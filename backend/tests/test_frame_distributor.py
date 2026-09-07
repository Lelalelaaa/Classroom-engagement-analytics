import asyncio

import numpy as np
import pytest

from app.pipeline import frame_distributor as fd
from app.pipeline.engagement_engine import AttentionConfig, AttentionState
from app.pipeline.head_pose import HeadPose
from app.pipeline.yunet_detector import FaceDetection


class _FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return list(self.detections)


@pytest.fixture
def pipeline(monkeypatch):
    """A distributor with the heavy models stubbed out, so tests run without weights."""
    detections = [
        FaceDetection(x=100, y=100, w=120, h=120, score=0.95),
        FaceDetection(x=600, y=120, w=110, h=110, score=0.92),
    ]
    monkeypatch.setattr(fd, "_detector", _FakeDetector(detections))
    monkeypatch.setattr(fd, "_get_detector", lambda: fd._detector)
    monkeypatch.setattr(fd, "_get_mp_pipeline", lambda: None)

    config = AttentionConfig(ema_alpha=1.0, window_seconds=10.0, calibration_seconds=2.0)
    return fd.FrameDistributor(max_faces=10, config=config), detections


def _frame():
    return np.zeros((1080, 1920, 3), np.uint8)


def test_missing_gaze_model_never_reports_engagement(monkeypatch, pipeline):
    """
    The bug this replaces: `g_score = gaze_res.is_looking_at_screen if gaze_res
    else True` made a missing model score every detected face as engaged, so
    the dashboard sat at ~88-100% no matter what the room was doing.
    """
    distributor, _ = pipeline
    monkeypatch.setattr(fd, "_gaze_model", None)
    monkeypatch.setattr(fd, "_gaze_load_failed", True)
    monkeypatch.setattr(distributor, "_analyse_face", lambda frame, det, want: (None, None))

    observations = asyncio.run(distributor.process_frame(_frame(), now=1.0))
    metrics = distributor.aggregate_class_metrics(observations, now=1.0, expected_count=2)

    assert len(observations) == 2
    assert all(o.state is AttentionState.UNKNOWN for o in observations)
    assert metrics["engaged_count"] == 0
    assert metrics["class_engagement"] == 0.0
    assert metrics["tracked_count"] == 0
    assert metrics["detected_count"] == 2
    assert metrics["gaze_model_loaded"] is False


def test_coverage_is_reported_not_hidden(monkeypatch, pipeline):
    """Detected-but-unmeasurable students must be visible as a coverage gap
    rather than quietly shrinking the denominator."""
    distributor, _ = pipeline
    monkeypatch.setattr(
        distributor, "_analyse_face",
        lambda frame, det, want: ((HeadPose(0.0, 0.0, 0.0), None) if det.x == 100 else (None, None)),
    )

    distributor.start_calibration(0.0)
    for t in (0.0, 0.5, 1.0, 1.5):
        asyncio.run(distributor.process_frame(_frame(), now=t))
    distributor.finish_calibration()

    observations = asyncio.run(distributor.process_frame(_frame(), now=3.0))
    metrics = distributor.aggregate_class_metrics(observations, now=3.0, expected_count=2)

    assert metrics["detected_count"] == 2
    assert metrics["tracked_count"] == 1
    assert metrics["expected_count"] == 2


def test_payload_carries_no_cross_frame_identifier(monkeypatch, pipeline):
    """
    Privacy invariant: the overlay may carry this frame's geometry, but
    nothing that lets a viewer follow one person through the session.
    """
    distributor, _ = pipeline
    monkeypatch.setattr(
        distributor, "_analyse_face", lambda frame, det, want: (HeadPose(0.0, 0.0, 0.0), None)
    )

    observations = asyncio.run(distributor.process_frame(_frame(), now=1.0))
    metrics = distributor.aggregate_class_metrics(observations, now=1.0)

    forbidden = {"track_id", "student_id", "face_id", "id", "name"}
    for face in metrics["faces"]:
        assert not forbidden & set(face)
    assert set(metrics["faces"][0]) == {
        "bbox", "yaw", "pitch", "state", "on_task_ratio", "measured", "has_gaze"
    }


def test_bboxes_are_normalised(monkeypatch, pipeline):
    distributor, detections = pipeline
    monkeypatch.setattr(
        distributor, "_analyse_face", lambda frame, det, want: (HeadPose(0.0, 0.0, 0.0), None)
    )
    observations = asyncio.run(distributor.process_frame(_frame(), now=1.0))
    for observation in observations:
        assert all(0.0 <= v <= 1.0 for v in observation.bbox_norm)


def test_no_detections_yields_empty_payload(monkeypatch, pipeline):
    distributor, _ = pipeline
    monkeypatch.setattr(fd, "_detector", _FakeDetector([]))
    monkeypatch.setattr(fd, "_get_detector", lambda: fd._detector)

    observations = asyncio.run(distributor.process_frame(_frame(), now=1.0))
    metrics = distributor.aggregate_class_metrics(observations, now=1.0)
    assert observations == []
    assert metrics["faces"] == []
    assert metrics["class_engagement"] == 0.0


def test_gaze_budget_limits_faces_per_frame(monkeypatch, pipeline):
    """Frame cost must stay flat as the class grows, not scale with attendance."""
    distributor, _ = pipeline
    many = [FaceDetection(x=100 + i * 150, y=100, w=120, h=120, score=0.9) for i in range(12)]
    monkeypatch.setattr(fd, "_detector", _FakeDetector(many))
    monkeypatch.setattr(fd, "_get_detector", lambda: fd._detector)
    monkeypatch.setattr(fd, "_get_gaze_model", lambda: object())
    monkeypatch.setattr(fd.settings, "GAZE_REFRESH_BUDGET", 4)

    tracked = distributor.tracker.update(many)
    assert len(distributor._select_gaze_targets(tracked)) == 4
