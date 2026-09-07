import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...config import settings
from ...services.alert_service import check_and_fire_alerts
from ...services.redis_service import get_redis, publish_metrics

router = APIRouter()


def _build_distributor():
    """
    One pipeline per session: the tracker, per-student calibration and
    attention history are session state and must not be shared between rooms.
    The heavy models behind it are process-wide singletons, so this is cheap.
    """
    from ...pipeline.frame_distributor import FrameDistributor
    return FrameDistributor(max_faces=settings.MAX_FACES_PER_CAMERA)


@router.websocket("/ws/video/{session_id}")
async def video_stream(websocket: WebSocket, session_id: str):
    """
    Receives JPEG frames from the classroom camera client, runs the gaze-only
    pipeline, and publishes ONLY anonymous aggregated metrics.

    Control messages (JSON text frames) drive calibration:
        {"action": "calibrate"} — start the "everyone look at the board" window

    PRIVACY GUARANTEE:
        - Frames are decoded in RAM as numpy arrays and released after use.
        - Raw video NEVER touches the filesystem or an external service.
        - The published payload carries class aggregates plus this frame's
          geometry — no identifiers, and nothing that links a face across frames.
    """
    await websocket.accept()
    redis = await get_redis()

    try:
        distributor = _build_distributor()
    except Exception as exc:
        await websocket.send_text(json.dumps({
            "error": f"AI pipeline unavailable: {exc}",
            "hint": "Run: python scripts/fetch_models.py",
        }))
        await websocket.close()
        return

    import cv2
    import numpy as np

    calibration_ends_at: float | None = None
    processing = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("text") is not None:
                calibration_ends_at = _handle_control(
                    message["text"], distributor, calibration_ends_at
                )
                continue

            raw_bytes = message.get("bytes")
            if not raw_bytes:
                continue

            # Backpressure: the camera keeps sending while a frame is in
            # flight. Dropping the newer frame is correct here — attention is
            # read over a multi-second window, so a skipped frame costs
            # nothing, whereas queueing them would make the dashboard lag
            # further behind the room with every frame.
            if processing:
                continue

            processing = True
            try:
                frame_array = np.frombuffer(raw_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                now = time.monotonic()
                if calibration_ends_at is not None and now >= calibration_ends_at:
                    calibrated = distributor.finish_calibration()
                    calibration_ends_at = None
                    print(f"[video_receiver] calibrated {calibrated} students in {session_id}")

                observations = await distributor.process_frame(frame_bgr, now)
                metrics = distributor.aggregate_class_metrics(observations, now)
                metrics["session_id"] = session_id

                # ── PRIVACY BOUNDARY: only the metrics dict crosses this line ──
                await publish_metrics(redis, session_id, metrics)
                if not metrics.get("is_calibrating"):
                    await check_and_fire_alerts(redis, session_id, metrics)

                del frame_bgr, frame_array
            finally:
                processing = False

    except (WebSocketDisconnect, RuntimeError):
        print(f"[video_receiver] Camera disconnected: session {session_id}")
    except Exception as exc:
        print(f"[video_receiver] Error in session {session_id}: {exc}")


def _handle_control(text: str, distributor, calibration_ends_at):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return calibration_ends_at
    if payload.get("action") == "calibrate":
        now = time.monotonic()
        distributor.start_calibration(now)
        print("[video_receiver] calibration started")
        return now + distributor.config.calibration_seconds
    return calibration_ends_at
