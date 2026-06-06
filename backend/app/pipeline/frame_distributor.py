"""
Full AI pipeline orchestrator.
Uses:
  - yakhyo/gaze-estimation  (ONNX) for gaze yaw/pitch angles
  - FER+ ONNX               for emotion classification
  - MediaPipe FaceLandmarker for yawn detection (mouth aspect ratio)
  - OpenCV Haarcascade      for fast face bounding box detection
"""
import asyncio
import logging
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dataclasses import dataclass, field

from .mediapipe_init import FacePipelineConfig
from .yawn_detector import YawnResult, detect_yawn
from .engagement_engine import FaceEngagementState, compute_engagement

logger   = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# Lazy-load the ONNX models so server starts even if weights are missing
_gaze_model    = None
_emotion_model = None
_face_cascade  = None


def _get_gaze_model():
    global _gaze_model
    if _gaze_model is None:
        try:
            from .gaze_onnx import GazeEstimatorONNX
            _gaze_model = GazeEstimatorONNX()
        except Exception as e:
            logger.warning(f"[pipeline] Gaze ONNX not available: {e}")
    return _gaze_model


def _get_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        try:
            from .emotion_onnx import EmotionClassifierONNX
            _emotion_model = EmotionClassifierONNX()
        except Exception as e:
            logger.warning(f"[pipeline] Emotion ONNX not available: {e}")
    return _emotion_model


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


# ── Per-face result (includes spatial data for frontend overlay) ─────────────

@dataclass
class FaceEngagementState:
    face_id:         int
    gaze_score:      float
    emotion_score:   float
    yawn_score:      float
    composite_score: float
    is_engaged:      bool
    # Overlay data (normalised 0–1 coordinates)
    bbox_norm:       list = field(default_factory=list)   # [x, y, w, h]
    yaw:             float = 0.0
    pitch:           float = 0.0
    emotion_label:   str   = "neutral"
    gaze_label:      str   = "Looking Forward"


class FrameDistributor:
    """
    Orchestrates the full AI pipeline for a single video frame.
    All processing is in-memory — no frames are written to disk.
    """

    def __init__(self, max_faces: int = 35):
        self.mp_pipeline   = FacePipelineConfig(max_faces=max_faces)
        self._emotion_cache: dict[int, object] = {}
        self._gaze_cache:   dict[int, object]  = {}

    async def process_frame(self, frame_bgr: np.ndarray) -> list[FaceEngagementState]:
        loop   = asyncio.get_running_loop()
        h, w   = frame_bgr.shape[:2]

        # ── Step 1: Face detection (OpenCV Haarcascade — fast) ────────────────
        gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        raw_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if not len(raw_faces):
            return []

        # ── Step 2: MediaPipe for yawn landmarks ──────────────────────────────
        mp_results = await loop.run_in_executor(
            executor, self.mp_pipeline.process_frame, frame_bgr
        )
        mp_landmarks = mp_results.multi_face_landmarks or []

        # ── Step 3: Per-face gaze + emotion (parallel) ───────────────────────
        states: list[FaceEngagementState] = []
        for face_id, (fx, fy, fw, fh) in enumerate(raw_faces):
            # Normalised bbox for frontend overlay
            bbox_norm = [fx / w, fy / h, fw / w, fh / h]

            # Crop face
            x1, y1 = max(0, fx), max(0, fy)
            x2, y2 = min(w, fx + fw), min(h, fy + fh)
            face_crop = frame_bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Gaze (ONNX — yakhyo model)
            try:
                gaze_res = await loop.run_in_executor(
                    executor, self._run_gaze, face_id, face_crop
                )
            except Exception as e:
                logger.warning(f"[pipeline] gaze error face {face_id}: {e}")
                gaze_res = None

            # Emotion (ONNX — FER+)
            try:
                em_res = await loop.run_in_executor(
                    executor, self._run_emotion, face_id, face_crop
                )
            except Exception as e:
                logger.warning(f"[pipeline] emotion error face {face_id}: {e}")
                em_res = None

            # Yawn (MediaPipe landmarks — match closest face)
            yawn_res = self._match_yawn(face_id, mp_landmarks, w, h, fx, fy, fw, fh)

            # Engagement score
            g_score  = (gaze_res.is_looking_at_screen if gaze_res else True)
            em_score = (em_res.valence if em_res else self._emotion_cache.get(face_id, 0.6) if not isinstance(self._emotion_cache.get(face_id), object) else 0.6)
            if em_res:
                em_score = em_res.valence
                self._emotion_cache[face_id] = em_res
            else:
                cached = self._emotion_cache.get(face_id)
                em_score = cached.valence if cached else 0.6

            y_score   = 0.0 if (yawn_res and yawn_res.is_yawning) else 1.0
            composite = round(0.5 * (1.0 if g_score else 0.0) + 0.3 * em_score + 0.2 * y_score, 3)

            states.append(FaceEngagementState(
                face_id=face_id,
                gaze_score=1.0 if g_score else 0.0,
                emotion_score=em_score,
                yawn_score=y_score,
                composite_score=composite,
                is_engaged=composite >= 0.5,
                bbox_norm=bbox_norm,
                yaw=gaze_res.yaw if gaze_res else 0.0,
                pitch=gaze_res.pitch if gaze_res else 0.0,
                emotion_label=em_res.dominant_emotion if em_res else "neutral",
                gaze_label=gaze_res.label if gaze_res else "👀 Looking Forward",
            ))

        return states

    def _run_gaze(self, face_id: int, face_crop: np.ndarray):
        model = _get_gaze_model()
        if model is None:
            return None
        result = model.estimate(face_crop)
        self._gaze_cache[face_id] = result
        return result

    def _run_emotion(self, face_id: int, face_crop: np.ndarray):
        model = _get_emotion_model()
        if model is None:
            return None
        result = model.classify(face_crop)
        if result:
            self._emotion_cache[face_id] = result
        return result or self._emotion_cache.get(face_id)

    def _match_yawn(self, face_id, mp_landmarks, w, h, fx, fy, fw, fh) -> Optional[YawnResult]:
        """Match the closest MediaPipe landmark set to the haarcascade face bbox."""
        if face_id < len(mp_landmarks):
            return detect_yawn(mp_landmarks[face_id], w, h)
        return None


def aggregate_class_metrics(states: list[FaceEngagementState]) -> dict:
    """
    Aggregate per-face states into anonymous class-level metrics.
    Also includes per-face spatial data for the frontend overlay.
    """
    if not states:
        return {
            "student_count":        0,
            "class_engagement":     0.0,
            "engaged_count":        0,
            "yawn_rate":            0.0,
            "emotion_distribution": {},
            "faces":                [],
        }

    n              = len(states)
    avg_engagement = sum(s.composite_score for s in states) / n
    yawn_rate      = sum(1 for s in states if s.yawn_score == 0.0) / n

    emotion_counts: dict[str, int] = {}
    for s in states:
        emotion_counts[s.emotion_label] = emotion_counts.get(s.emotion_label, 0) + 1

    # Per-face data for camera overlay (no identifiers — geometry only)
    faces_overlay = [
        {
            "bbox":    s.bbox_norm,
            "yaw":     s.yaw,
            "pitch":   s.pitch,
            "emotion": s.emotion_label,
            "gaze":    s.gaze_label,
            "engaged": s.is_engaged,
            "score":   s.composite_score,
        }
        for s in states
    ]

    return {
        "student_count":        n,
        "class_engagement":     round(avg_engagement * 100, 1),
        "engaged_count":        sum(1 for s in states if s.is_engaged),
        "yawn_rate":            round(yawn_rate * 100, 1),
        "emotion_distribution": emotion_counts,
        "faces":                faces_overlay,
    }
