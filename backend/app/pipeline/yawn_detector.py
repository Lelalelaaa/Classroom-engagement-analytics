from dataclasses import dataclass
import numpy as np

# Inner lip landmarks for MAR calculation
MOUTH_TOP_INDICES    = [13, 14]
MOUTH_BOTTOM_INDICES = [17, 18]
MOUTH_LEFT           = 78
MOUTH_RIGHT          = 308

MAR_YAWN_THRESHOLD = 0.6   # Tunable: increase if false positives on wide mouths


@dataclass
class YawnResult:
    is_yawning: bool
    mar_score: float


def detect_yawn(landmarks, img_w: int, img_h: int) -> YawnResult:
    """
    Mouth Aspect Ratio (MAR):
        MAR = vertical_mouth_opening / horizontal_mouth_width
    MAR > threshold  →  yawning detected.
    """

    def px(idx: int) -> np.ndarray:
        lm = landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    top    = np.mean([px(i) for i in MOUTH_TOP_INDICES],    axis=0)
    bottom = np.mean([px(i) for i in MOUTH_BOTTOM_INDICES], axis=0)
    left   = px(MOUTH_LEFT)
    right  = px(MOUTH_RIGHT)

    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right) + 1e-6
    mar        = vertical / horizontal

    return YawnResult(is_yawning=mar > MAR_YAWN_THRESHOLD, mar_score=round(mar, 3))
