from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...services.redis_service import publish_metrics, get_redis
from ...services.alert_service import check_and_fire_alerts
from ...config import settings

router = APIRouter()

# Lazy-load heavy AI dependencies so the server starts even without them
_distributor = None

def _get_distributor():
    global _distributor
    if _distributor is None:
        try:
            from ...pipeline.frame_distributor import FrameDistributor
            _distributor = FrameDistributor(max_faces=settings.MAX_FACES_PER_CAMERA)
            print("[video_receiver] AI pipeline ready (gaze + emotion + yawn)")
        except Exception as e:
            print(f"[video_receiver] AI pipeline unavailable: {e}")
            _distributor = None
    return _distributor


@router.websocket("/ws/video/{session_id}")
async def video_stream(websocket: WebSocket, session_id: str):
    """
    Receives raw JPEG frames from the classroom camera client.
    Runs full AI pipeline and publishes ONLY anonymous aggregated metrics.

    PRIVACY GUARANTEE:
        - Frames are decoded in RAM as numpy arrays.
        - They are explicitly deleted (del) after processing.
        - Raw video NEVER touches the filesystem or external services.
    """
    await websocket.accept()
    redis = await get_redis()
    distributor = _get_distributor()

    if distributor is None:
        await websocket.send_text('{"error": "AI pipeline not available. Install: numpy opencv-python-headless mediapipe"}')
        await websocket.close()
        return

    try:
        import numpy as np
        import cv2
        from ...pipeline.frame_distributor import aggregate_class_metrics

        while True:
            raw_bytes   = await websocket.receive_bytes()
            frame_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame_bgr   = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                continue

            face_states = await distributor.process_frame(frame_bgr)
            metrics     = aggregate_class_metrics(face_states)
            metrics["session_id"] = session_id

            # ── PRIVACY BOUNDARY: only metrics dict crosses this line ──
            await publish_metrics(redis, session_id, metrics)
            await check_and_fire_alerts(redis, session_id, metrics)

            # Explicit frame cleanup — frames never persist
            del frame_bgr, frame_array, raw_bytes

    except WebSocketDisconnect:
        print(f"[video_receiver] Camera disconnected: session {session_id}")
    except Exception as exc:
        print(f"[video_receiver] Error in session {session_id}: {exc}")
