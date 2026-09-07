import pytest

from app.pipeline.engagement_engine import (
    AttentionConfig, AttentionEngine, AttentionState,
)
from app.pipeline.head_pose import HeadPose


def _config(**overrides) -> AttentionConfig:
    base = dict(
        yaw_enter_deg=25.0, yaw_exit_deg=32.0, desk_work_pitch_deg=18.0,
        desk_work_weight=1.0, ema_alpha=1.0, window_seconds=10.0,
        on_task_floor=0.5, calibration_seconds=2.0,
    )
    base.update(overrides)
    return AttentionConfig(**base)


def _calibrate(engine, poses: dict[int, HeadPose], start=0.0):
    engine.start_calibration(start)
    for t in (start, start + 0.5, start + 1.0, start + 1.5):
        for track_id, pose in poses.items():
            engine.observe(track_id, t, pose, face_px=120)
    return engine.finish_calibration()


def test_calibration_cancels_seat_position():
    """
    The central claim of the model. Two students in different seats face the
    board with very different camera-relative angles; an absolute cone around
    the camera axis would score the front-left one as distracted.
    """
    engine = AttentionEngine(_config())
    front_left = HeadPose(yaw=-38.0, pitch=2.0, roll=0.0)
    far_right = HeadPose(yaw=31.0, pitch=6.0, roll=0.0)
    assert _calibrate(engine, {0: front_left, 1: far_right}) == 2

    engine.observe(0, 3.0, front_left, face_px=120)
    engine.observe(1, 3.0, far_right, face_px=120)
    assert engine.students[0].state is AttentionState.ON_TASK
    assert engine.students[1].state is AttentionState.ON_TASK


def test_looking_down_is_desk_work_not_off_task():
    """A student reading their notes is on task; scoring them off-task is the
    false negative a teacher would reject outright."""
    engine = AttentionEngine(_config())
    reference = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
    _calibrate(engine, {0: reference})

    engine.observe(0, 3.0, HeadPose(yaw=2.0, pitch=30.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.DESK_WORK


def test_turning_away_is_off_task():
    engine = AttentionEngine(_config())
    _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})
    engine.observe(0, 3.0, HeadPose(yaw=55.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.OFF_TASK


def test_hysteresis_keeps_a_borderline_student_stable():
    """Between the enter and exit thresholds the previous state must persist,
    or students flap between engaged and not on every frame."""
    engine = AttentionEngine(_config())
    _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})

    engine.observe(0, 3.0, HeadPose(yaw=5.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.ON_TASK

    # 28 deg: past the 25 enter threshold but inside the 32 exit threshold
    engine.observe(0, 3.5, HeadPose(yaw=28.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.ON_TASK

    engine.observe(0, 4.0, HeadPose(yaw=40.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.OFF_TASK

    # coming back, it now takes the stricter enter threshold to regain on-task
    engine.observe(0, 4.5, HeadPose(yaw=28.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[0].state is AttentionState.OFF_TASK


def test_unmeasurable_students_are_never_scored_engaged():
    """
    The regression that motivated this rewrite: a missing measurement used to
    fall back to `True`, so the dashboard reported ~88-100% engagement for any
    detected face regardless of behaviour.
    """
    engine = AttentionEngine(_config())
    _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})

    engine.observe(0, 3.0, head_pose=None, face_px=120)
    assert engine.students[0].state is AttentionState.UNKNOWN

    engine.observe(0, 3.5, HeadPose(0.0, 0.0, 0.0), face_px=8)
    assert engine.students[0].state is AttentionState.UNKNOWN

    metrics = engine.class_metrics(now=3.5)
    assert metrics["engaged_count"] == 0
    assert metrics["tracked_count"] == 0
    assert metrics["class_engagement"] == 0.0


def test_unknown_samples_are_excluded_from_the_ratio():
    """Unmeasured time must not be averaged in as if it were disengagement."""
    engine = AttentionEngine(_config())
    _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})

    engine.observe(0, 3.0, HeadPose(0.0, 0.0, 0.0), face_px=120)
    engine.observe(0, 3.5, head_pose=None, face_px=120)
    engine.observe(0, 4.0, HeadPose(0.0, 0.0, 0.0), face_px=120)

    ratio = engine.students[0].on_task_ratio(4.0, window_seconds=10.0, desk_work_weight=1.0)
    assert ratio == pytest.approx(1.0)


def test_uncalibrated_late_arrival_falls_back_to_the_class_reference():
    engine = AttentionEngine(_config())
    _calibrate(engine, {0: HeadPose(20.0, 0.0, 0.0), 1: HeadPose(24.0, 0.0, 0.0)})

    engine.observe(9, 3.0, HeadPose(yaw=22.0, pitch=0.0, roll=0.0), face_px=120)
    assert engine.students[9].reference is None
    assert engine.students[9].state is AttentionState.ON_TASK


def test_below_floor_count_surfaces_a_disengaged_minority():
    """A plain mean hides the case a teacher most needs to see."""
    engine = AttentionEngine(_config(window_seconds=10.0))
    poses = {i: HeadPose(0.0, 0.0, 0.0) for i in range(4)}
    _calibrate(engine, poses)

    for t in (3.0, 3.5, 4.0, 4.5):
        for i in range(3):
            engine.observe(i, t, HeadPose(0.0, 0.0, 0.0), face_px=120)
        engine.observe(3, t, HeadPose(60.0, 0.0, 0.0), face_px=120)

    metrics = engine.class_metrics(now=4.5, expected_count=4)
    assert metrics["below_floor_count"] == 1
    assert metrics["tracked_count"] == 4
    assert metrics["expected_count"] == 4


def test_desk_work_weight_controls_how_much_it_counts():
    """Whether desk work counts as engagement is fitted from teacher feedback,
    so the weight has to actually move the ratio."""
    for weight, expected in ((1.0, 1.0), (0.0, 0.0), (0.5, 0.5)):
        engine = AttentionEngine(_config(desk_work_weight=weight))
        _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})
        engine.observe(0, 3.0, HeadPose(yaw=0.0, pitch=30.0, roll=0.0), face_px=120)
        ratio = engine.students[0].on_task_ratio(3.0, 10.0, weight)
        assert ratio == pytest.approx(expected)


def test_eye_gaze_is_ignored_below_the_pixel_gate():
    """Behind the gate the gaze model reads upsampling artifacts, so a wild
    value from it must not move the verdict."""
    engine = AttentionEngine(_config(min_face_px_for_gaze=80, gaze_blend_weight=1.0))
    _calibrate(engine, {0: HeadPose(0.0, 0.0, 0.0)})

    engine.observe(0, 3.0, HeadPose(0.0, 0.0, 0.0), gaze=(90.0, 0.0), face_px=40)
    assert engine.students[0].state is AttentionState.ON_TASK

    engine.observe(0, 3.5, HeadPose(0.0, 0.0, 0.0), gaze=(90.0, 0.0), face_px=120)
    assert engine.students[0].state is AttentionState.OFF_TASK
