"""
Eye-gaze estimation with the yakhyo/gaze-estimation ONNX models (L2CS-style
binned classification over 90 bins of 4 degrees, offset by 180).

Scope note: this model contributes only for faces above
`settings.MIN_FACE_PX_FOR_GAZE`. Behind that gate the eye region is a handful
of pixels and the model is reading interpolation artifacts, so the pipeline
falls back to head pose alone rather than emitting a confident guess.

This module returns ANGLES ONLY. Deciding whether a student is attentive is
the engagement engine's job, because that decision needs the student's own
calibrated reference — a camera-relative "is the face pointed at the lens"
test scores a front-row student looking at the board as distracted.

Reference: https://github.com/yakhyo/gaze-estimation
"""
import os
import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .yunet_detector import FaceDetection

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "gaze-estimation", "weights")
)


@dataclass(frozen=True)
class GazeResult:
    """Camera-relative eye gaze, in degrees."""
    yaw: float
    pitch: float


class GazeEstimatorONNX:
    _BINS      = 90
    _BINWIDTH  = 4
    _ANGLE_OFF = 180
    _INPUT     = 448
    _MEAN      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD       = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _IDX       = np.arange(_BINS, dtype=np.float32)

    def __init__(self, backbone: str = "resnet34", margin: float = 0.28, intra_op_threads: int = 4):
        path = os.path.join(_WEIGHTS_DIR, f"{backbone}_gaze.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Gaze ONNX model not found at {path}. Run: python scripts/fetch_models.py"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, intra_op_threads)
        self._session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._margin = margin
        logger.info("[gaze_onnx] loaded %s_gaze.onnx", backbone)

    def estimate(self, frame_bgr: np.ndarray, det: FaceDetection) -> Optional[GazeResult]:
        """Estimate gaze for one detected face. Returns None if the crop is unusable."""
        crop = self._aligned_crop(frame_bgr, det)
        if crop is None:
            return None
        yaw_logits, pitch_logits = self._session.run(
            self._output_names, {self._input_name: self._preprocess(crop)}
        )
        return GazeResult(
            yaw=round(self._decode(yaw_logits), 2),
            pitch=round(self._decode(pitch_logits), 2),
        )

    def _aligned_crop(self, frame_bgr: np.ndarray, det: FaceDetection) -> Optional[np.ndarray]:
        """
        Square, roll-corrected crop with margin around the detection.

        The model is trained on margined, eye-level-aligned faces. Feeding it
        the tight, unrotated YuNet box costs accuracy for no reason, and the
        margin has to be taken from the full frame — it cannot be recovered
        from a crop that was already cut to the box.
        """
        h, w = frame_bgr.shape[:2]
        side = max(det.w, det.h) * (1.0 + 2.0 * self._margin)
        if side < 2:
            return None
        cx, cy = det.x + det.w / 2.0, det.y + det.h / 2.0

        angle = self._roll_degrees(det)
        out = int(round(side))
        mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mat[0, 2] += out / 2.0 - cx
        mat[1, 2] += out / 2.0 - cy

        crop = cv2.warpAffine(
            frame_bgr, mat, (out, out),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return crop if crop.size else None

    @staticmethod
    def _roll_degrees(det: FaceDetection) -> float:
        """In-plane rotation from the eye landmarks; 0 if YuNet gave none."""
        lm = det.landmarks
        if lm is None or lm.shape != (5, 2):
            return 0.0
        right_eye, left_eye = lm[0], lm[1]
        dx, dy = float(left_eye[0] - right_eye[0]), float(left_eye[1] - right_eye[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._INPUT, self._INPUT)).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        return np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)

    def _decode(self, logits: np.ndarray) -> float:
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
        return float(np.sum(probs * self._IDX, axis=1)[0] * self._BINWIDTH - self._ANGLE_OFF)
