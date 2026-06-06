"""
Gaze estimation using yakhyo/gaze-estimation ONNX model (resnet18_gaze.onnx).
Returns (yaw, pitch) in radians — no PyTorch needed, pure ONNX Runtime.

Reference: https://github.com/yakhyo/gaze-estimation
"""
import os
import numpy as np
import cv2
import onnxruntime as ort
from dataclasses import dataclass

GAZE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "models", "gaze-estimation", "weights", "resnet18_gaze.onnx"
)

# Gaze direction labels based on yaw angle
GAZE_LABELS = {
    "forward":      "👀 Looking Forward",
    "left":         "← Looking Left",
    "right":        "→ Looking Right",
    "down":         "↓ Looking Down",
    "up":           "↑ Looking Up",
}


@dataclass
class GazeResult:
    """Gaze estimation result from ONNX model."""
    yaw:   float   # horizontal angle in radians (negative = left, positive = right)
    pitch: float   # vertical angle in radians (negative = up, positive = down)
    label: str     # human-readable direction
    is_looking_at_screen: bool   # True if roughly forward-facing
    gaze_ratio: float            # 0=hard left, 1=hard right, ~0.5=center


class GazeEstimatorONNX:
    """
    Wraps yakhyo/gaze-estimation ONNX model.
    Input:  BGR face crop (any size) → resized to 448×448
    Output: (yaw, pitch) in radians
    """

    _BINS       = 90
    _BINWIDTH   = 4
    _ANGLE_OFF  = 180
    _MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _IDX        = np.arange(90, dtype=np.float32)

    def __init__(self):
        path = os.path.abspath(GAZE_MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Gaze ONNX model not found at {path}. "
                "Download from: https://github.com/yakhyo/gaze-estimation/releases"
            )
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        print("[gaze_onnx] Loaded resnet18_gaze.onnx [OK]")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448)).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        return np.expand_dims(img.transpose(2, 0, 1), 0)   # BCHW

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def estimate(self, face_bgr: np.ndarray) -> GazeResult:
        """Run gaze estimation on a face crop. Returns GazeResult."""
        inp = self._preprocess(face_bgr)
        yaw_logits, pitch_logits = self._session.run(
            self._output_names, {self._input_name: inp}
        )
        yaw_deg   = float(np.sum(self._softmax(yaw_logits)   * self._IDX, axis=1) * self._BINWIDTH - self._ANGLE_OFF)
        pitch_deg = float(np.sum(self._softmax(pitch_logits) * self._IDX, axis=1) * self._BINWIDTH - self._ANGLE_OFF)

        yaw_rad   = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)

        # Normalise yaw to 0-1 ratio (like the original iris approach)
        gaze_ratio = max(0.0, min(1.0, (yaw_deg + 30) / 60))   # ±30° maps to 0-1

        # Classify direction
        if abs(yaw_deg) < 15 and abs(pitch_deg) < 15:
            label  = "forward"
            looking = True
        elif yaw_deg < -15:
            label  = "left";  looking = False
        elif yaw_deg > 15:
            label  = "right"; looking = False
        elif pitch_deg < -15:
            label  = "up";    looking = False
        else:
            label  = "down";  looking = False

        return GazeResult(
            yaw=round(yaw_rad, 3),
            pitch=round(pitch_rad, 3),
            label=GAZE_LABELS[label],
            is_looking_at_screen=looking,
            gaze_ratio=round(gaze_ratio, 3),
        )
