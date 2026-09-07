"""
MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10).
Auto-downloads face_landmarker.task on first run (~6 MB).

Two modes, both needed:

  process_crop()  — the live path. YuNet finds the faces on the full frame and
                    this runs on each crop, where the face fills the input and
                    the landmarker's short-range detector is inside its
                    comfortable range.
  process_frame() — full-frame, multi-face. Kept so the eval harness can
                    measure landmarker-alone recall against YuNet+crop and
                    settle the architecture on numbers rather than assumption.

Both request `output_facial_transformation_matrixes`, which is what makes head
pose available at zero extra inference cost (see head_pose.pose_from_matrix).
"""
import os
import queue
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
    """Adapts the Tasks API result to the shape the rest of the pipeline reads."""
    def __init__(self, task_result):
        if task_result.face_landmarks:
            self.multi_face_landmarks = [
                _FaceLandmarksCompat(lm_list) for lm_list in task_result.face_landmarks
            ]
        else:
            self.multi_face_landmarks = None

        self.facial_transformation_matrixes = list(
            getattr(task_result, "facial_transformation_matrixes", None) or []
        )

    @property
    def face_count(self) -> int:
        return len(self.multi_face_landmarks or [])


class FacePipelineConfig:
    """Shared FaceLandmarker instances reused across all frames."""

    def __init__(self, max_faces: int = 40, crop_pool_size: int = 4):
        _ensure_model()
        self._landmarker = self._build(max_faces)
        # A FaceLandmarker cannot serve concurrent detect() calls, so the
        # worker pool borrows one instance each rather than sharing one.
        self._crop_pool: queue.Queue = queue.Queue()
        for _ in range(max(1, crop_pool_size)):
            self._crop_pool.put(self._build(1))
        print(f"[mediapipe] FaceLandmarker ready (max_faces={max_faces}, "
              f"crop_pool={crop_pool_size}, +transform matrices)")

    @staticmethod
    def _build(num_faces: int):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> _ResultCompat:
        return _ResultCompat(self._landmarker.detect(_to_mp_image(frame_bgr)))

    def process_crop(self, crop_bgr: np.ndarray) -> _ResultCompat:
        """Landmark a single pre-detected face crop. Safe to call from any worker."""
        landmarker = self._crop_pool.get()
        try:
            return _ResultCompat(landmarker.detect(_to_mp_image(crop_bgr)))
        finally:
            self._crop_pool.put(landmarker)


def _to_mp_image(frame_bgr: np.ndarray) -> "mp.Image":
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
