from dataclasses import dataclass
from typing import Optional

from .gaze_estimator import GazeResult
from .emotion_classifier import EmotionResult
from .yawn_detector import YawnResult

# Emotion → engagement valence map
EMOTION_VALENCE: dict[str, float] = {
    "happy":     1.0,
    "surprised": 0.8,
    "neutral":   0.6,
    "sad":       0.3,
    "angry":     0.2,
    "fear":      0.2,
    "disgust":   0.1,
}

# Signal weights — must sum to 1.0
WEIGHTS: dict[str, float] = {
    "gaze":    0.50,
    "emotion": 0.30,
    "yawn":    0.20,
}

ENGAGED_THRESHOLD = 0.5


@dataclass
class FaceEngagementState:
    face_id:         int
    gaze_score:      float   # 0.0 or 1.0 (binary)
    emotion_score:   float   # 0.0 – 1.0 (valence)
    yawn_score:      float   # 1.0 = not yawning, 0.0 = yawning
    composite_score: float   # Weighted fusion: 0.0 – 1.0
    is_engaged:      bool


def compute_engagement(
    face_id: int,
    gaze: GazeResult,
    emotion: Optional[EmotionResult],
    yawn: YawnResult,
    last_emotion_score: float = 0.6,
) -> FaceEngagementState:
    """
    Fuse gaze, emotion, and yawn signals into a single Engagement Score.

    Formula:
        E = (W_gaze × G) + (W_emotion × Em) + (W_yawn × Y)

    Where:
        G  = 1.0 if is_looking_at_screen else 0.0
        Em = EMOTION_VALENCE[dominant_emotion]  (or last cached value)
        Y  = 1.0 if not yawning, else 0.0
    """
    g_score  = 1.0 if gaze.is_looking_at_screen else 0.0
    em_score = (
        EMOTION_VALENCE.get(emotion.dominant_emotion, 0.6)
        if emotion
        else last_emotion_score
    )
    y_score  = 0.0 if yawn.is_yawning else 1.0

    composite = (
        WEIGHTS["gaze"]    * g_score  +
        WEIGHTS["emotion"] * em_score +
        WEIGHTS["yawn"]    * y_score
    )

    return FaceEngagementState(
        face_id=face_id,
        gaze_score=g_score,
        emotion_score=em_score,
        yawn_score=y_score,
        composite_score=round(composite, 3),
        is_engaged=composite >= ENGAGED_THRESHOLD,
    )
