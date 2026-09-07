import numpy as np
import pytest

from app.pipeline.head_pose import HeadPose, angular_deviation, pose_from_matrix


def _rx(deg):
    r = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])


def _ry(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])


def _rz(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])


def _matrix(rotation, scale=1.0):
    m = np.eye(4)
    m[:3, :3] = rotation * scale
    return m


def test_yaw_is_recovered():
    pose = pose_from_matrix(_matrix(_ry(20)))
    assert pose.yaw == pytest.approx(20.0, abs=0.01)
    assert pose.pitch == pytest.approx(0.0, abs=0.01)


def test_looking_down_is_positive_pitch():
    """The desk-work state depends on this sign; a flip would invert it."""
    pose = pose_from_matrix(_matrix(_rx(-15)))
    assert pose.pitch > 0


def test_scale_is_removed_before_decomposition():
    """MediaPipe's matrix carries distance as scale — angles must not track it."""
    near = pose_from_matrix(_matrix(_rz(10) @ _ry(20) @ _rx(15), scale=1.0))
    far = pose_from_matrix(_matrix(_rz(10) @ _ry(20) @ _rx(15), scale=8.3))
    assert near.yaw == pytest.approx(far.yaw, abs=0.01)
    assert near.pitch == pytest.approx(far.pitch, abs=0.01)


def test_malformed_input_returns_none():
    assert pose_from_matrix(None) is None
    assert pose_from_matrix(np.zeros((4, 4))) is None
    assert pose_from_matrix(np.eye(3)) is None


def test_deviation_is_relative_to_reference():
    pose = HeadPose(yaw=30.0, pitch=5.0, roll=0.0)
    reference = HeadPose(yaw=25.0, pitch=3.0, roll=0.0)
    yaw_dev, pitch_dev = angular_deviation(pose, reference)
    assert yaw_dev == pytest.approx(5.0)
    assert pitch_dev == pytest.approx(2.0)


def test_deviation_wraps_the_short_way():
    yaw_dev, _ = angular_deviation(HeadPose(-170.0, 0, 0), HeadPose(170.0, 0, 0))
    assert yaw_dev == pytest.approx(20.0)
