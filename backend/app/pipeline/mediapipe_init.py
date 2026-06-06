"""
MediaPipe FaceLandmarker using the new Tasks API (mediapipe >= 0.10).
Wraps the new result format to stay compatible with the rest of the pipeline.
Auto-downloads the face_landmarker.task model on first run (~6 MB).
"""
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("[mediapipe] Downloading face_landmarker.task (~6 MB) …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[mediapipe] Model downloaded and ready.")


# ── Compatibility wrappers so gaze/yawn estimators don't need changes ─────────

class _LandmarkCompat:
    """Makes a single NormalizedLandmark accessible as .x .y .z"""
    __slots__ = ("x", "y", "z")

    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z


class _FaceLandmarksCompat:
    """Wraps a list of NormalizedLandmarks as .landmark[idx]"""
    def __init__(self, lm_list):
        self.landmark = [_LandmarkCompat(lm) for lm in lm_list]


class _ResultCompat:
    """Makes new Tasks result look like old mp.solutions result."""
    def __init__(self, task_result):
        if task_result.face_landmarks:
            self.multi_face_landmarks = [
                _FaceLandmarksCompat(lm_list)
                for lm_list in task_result.face_landmarks
            ]
        else:
            self.multi_face_landmarks = None


# ── Main pipeline class ───────────────────────────────────────────────────────

class FacePipelineConfig:
    """
    Shared FaceLandmarker instance reused across all frames.
    Compatible with mediapipe >= 0.10 (Tasks API).
    """

    def __init__(self, max_faces: int = 35):
        _ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print(f"[mediapipe] FaceLandmarker ready (max_faces={max_faces})")

    def process_frame(self, frame_bgr: np.ndarray) -> _ResultCompat:
        """Run face landmark detection and return a compatibility-wrapped result."""
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)
        return _ResultCompat(result)
