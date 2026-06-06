from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

EmotionLabel = Literal["happy", "neutral", "sad", "surprised", "angry", "fear", "disgust"]


@dataclass
class EmotionResult:
    dominant_emotion: EmotionLabel
    scores: dict = field(default_factory=dict)


# Module-level frame counter for skip logic (per-process, not per-face)
_frame_counter: int = 0
EMOTION_SKIP_FRAMES: int = 5  # Run DeepFace every N frames to manage CPU load


def classify_emotion(face_roi: np.ndarray) -> Optional[EmotionResult]:
    """
    Classify emotion from a cropped face ROI (BGR numpy array).
    Returns None on skipped frames so callers use their cached result.
    """
    global _frame_counter
    _frame_counter += 1

    if _frame_counter % EMOTION_SKIP_FRAMES != 0:
        return None  # Caller should use cached value

    try:
        from deepface import DeepFace  # Lazy import: heavy startup cost
        results = DeepFace.analyze(
            face_roi,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        emotions  = results[0]["emotion"]
        dominant  = results[0]["dominant_emotion"]
        return EmotionResult(dominant_emotion=dominant, scores=emotions)
    except Exception:
        return EmotionResult(dominant_emotion="neutral", scores={})
