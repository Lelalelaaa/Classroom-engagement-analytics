"""
Emotion classification using FER+ ONNX model (emotion-ferplus-8.onnx).
No TensorFlow required — pure ONNX Runtime.

Model: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus
Input:  [1, 1, 64, 64]  grayscale float32
Output: [1, 8]           logits for 8 emotion classes
"""
import os
import numpy as np
import cv2
import onnxruntime as ort
from dataclasses import dataclass, field

EMOTION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "emotion-ferplus-8.onnx"
)

# FER+ label order
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

# Map FER+ labels → ReLi engagement valence (0.0–1.0)
EMOTION_VALENCE = {
    "happiness":  1.0,
    "surprise":   0.75,
    "neutral":    0.6,
    "contempt":   0.4,
    "sadness":    0.3,
    "fear":       0.2,
    "anger":      0.2,
    "disgust":    0.1,
}

# Emoji for each emotion
EMOTION_EMOJI = {
    "happiness": "😊", "surprise": "😮", "neutral": "😐",
    "sadness": "😢",   "anger": "😠",    "disgust": "🤢",
    "fear": "😨",      "contempt": "😒",
}

_skip_counter = 0
EMOTION_SKIP_FRAMES = 5


@dataclass
class EmotionResult:
    dominant_emotion: str
    scores: dict = field(default_factory=dict)
    valence: float = 0.6

    @property
    def emoji(self) -> str:
        return EMOTION_EMOJI.get(self.dominant_emotion, "😐")


class EmotionClassifierONNX:
    """
    oarriaga-style FER emotion classifier backed by FER+ ONNX.
    Compatible with the rest of the ReLi pipeline.
    """

    def __init__(self):
        path = os.path.abspath(EMOTION_MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Emotion ONNX model not found at {path}. "
                "Download from: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus"
            )
        self._session     = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        print("[emotion_onnx] Loaded emotion-ferplus-8.onnx [OK]")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32)
        return gray.reshape(1, 1, 64, 64)   # BCHW

    def classify(self, face_bgr: np.ndarray) -> EmotionResult | None:
        """
        Classify emotion from a face crop. Returns None on skip frames
        so callers can use their cached result.
        """
        global _skip_counter
        _skip_counter += 1
        if _skip_counter % EMOTION_SKIP_FRAMES != 0:
            return None

        try:
            inp     = self._preprocess(face_bgr)
            logits  = self._session.run([self._output_name], {self._input_name: inp})[0][0]
            probs   = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
            idx     = int(np.argmax(probs))
            label   = EMOTION_LABELS[idx]
            scores  = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
            return EmotionResult(
                dominant_emotion=label,
                scores=scores,
                valence=EMOTION_VALENCE.get(label, 0.6),
            )
        except Exception as e:
            print(f"[emotion_onnx] classify error: {e}")
            return EmotionResult(dominant_emotion="neutral", valence=0.6)
