from dataclasses import dataclass
import numpy as np

# MediaPipe FaceMesh landmark indices (refine_landmarks=True required for iris)
LEFT_EYE_CORNERS  = (33, 133)    # outer=33, inner=133
RIGHT_EYE_CORNERS = (362, 263)   # inner=362, outer=263
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473


@dataclass
class GazeResult:
    face_id: int
    is_looking_at_screen: bool
    gaze_ratio: float   # 0.0=far left, 1.0=far right, ~0.5=centred


def estimate_gaze(
    landmarks,
    img_w: int,
    img_h: int,
    threshold: float = 0.25,
) -> GazeResult:
    """
    Compute gaze ratio by comparing iris centre to eye-corner positions.
    A ratio close to 0.5 means the student is looking forward (at the board/screen).
    """

    def get_px(idx: int) -> np.ndarray:
        lm = landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    # Left eye
    l_outer, l_inner = get_px(LEFT_EYE_CORNERS[0]), get_px(LEFT_EYE_CORNERS[1])
    l_iris = get_px(LEFT_IRIS_CENTER)
    l_ratio = (l_iris[0] - l_outer[0]) / (l_inner[0] - l_outer[0] + 1e-6)

    # Right eye
    r_inner, r_outer = get_px(RIGHT_EYE_CORNERS[0]), get_px(RIGHT_EYE_CORNERS[1])
    r_iris = get_px(RIGHT_IRIS_CENTER)
    r_ratio = (r_iris[0] - r_outer[0]) / (r_inner[0] - r_outer[0] + 1e-6)

    avg_ratio = (l_ratio + r_ratio) / 2.0
    is_centered = abs(avg_ratio - 0.5) < threshold

    return GazeResult(face_id=0, is_looking_at_screen=is_centered, gaze_ratio=round(avg_ratio, 3))
