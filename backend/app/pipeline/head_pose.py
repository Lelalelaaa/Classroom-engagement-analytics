"""
Head pose from the MediaPipe FaceLandmarker facial transformation matrix.

At classroom distance head pose — not eye gaze — is the recoverable signal. A
face 6 m from a 1080p camera is ~40-60 px wide, which leaves an eye region of
~8-12 px and an iris of 2-3 px; iris-based yaw cannot be recovered from that.
Head pose stays usable at those sizes and costs nothing extra, since the
landmarker already computes the matrix (see mediapipe_init.FacePipelineConfig).

Sign convention used throughout the pipeline:
    yaw   > 0  → subject turned toward the camera's right
    pitch > 0  → subject looking DOWN   (this is what separates desk-work from off-task)
    roll  > 0  → subject's head tilted clockwise in the image
"""
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


def pose_from_matrix(matrix) -> Optional[HeadPose]:
    """
    Decompose a 4x4 facial transformation matrix into Euler angles in degrees.

    The matrix carries scale as well as rotation, so each column of the
    rotation block is normalised before decomposition — skipping this makes
    RQDecomp3x3 return angles that drift with the subject's distance.
    """
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        return None

    rot = m[:3, :3].copy()
    norms = np.linalg.norm(rot, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms < 1e-8):
        return None
    rot /= norms

    try:
        euler = cv2.RQDecomp3x3(rot)[0]
    except cv2.error as exc:
        logger.debug("[head_pose] RQDecomp3x3 failed: %s", exc)
        return None

    x_deg, y_deg, z_deg = (float(v) for v in euler)

    # MediaPipe's camera space is Y-up / Z-toward-viewer, so a downward head
    # tilt comes back as a negative rotation about X. Flip it so that
    # "positive pitch means looking down" holds for the desk-work rule.
    # TODO: assumed sign convention — confirm against the first annotated
    # recording (eval/verify_pose_signs.py) before trusting desk-work labels.
    return HeadPose(
        yaw=_wrap180(y_deg),
        pitch=_wrap180(-x_deg),
        roll=_wrap180(z_deg),
    )


def _wrap180(deg: float) -> float:
    """Map an angle into (-180, 180] so that near-frontal poses stay near zero."""
    wrapped = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def angular_deviation(pose: HeadPose, reference: HeadPose) -> tuple[float, float]:
    """
    Yaw and pitch of `pose` relative to a student's own calibrated `reference`.

    Working in relative terms is what makes the model viable: it cancels seat
    position, the camera-vs-board offset, and the per-subject bias of the pose
    estimator in one step. None of those can be addressed by an absolute
    threshold on the camera-relative angle.
    """
    return _wrap180(pose.yaw - reference.yaw), _wrap180(pose.pitch - reference.pitch)
