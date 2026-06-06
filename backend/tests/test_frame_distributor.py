import pytest
from app.pipeline.frame_distributor import aggregate_class_metrics
from app.pipeline.engagement_engine import FaceEngagementState


def make_state(face_id, composite, is_engaged, yawn_score=1.0, em_score=0.6):
    return FaceEngagementState(
        face_id=face_id,
        gaze_score=1.0,
        emotion_score=em_score,
        yawn_score=yawn_score,
        composite_score=composite,
        is_engaged=is_engaged,
    )


def test_aggregate_empty():
    result = aggregate_class_metrics([])
    assert result["student_count"] == 0
    assert result["class_engagement"] == 0.0


def test_aggregate_all_engaged():
    states = [make_state(i, 0.85, True) for i in range(5)]
    result = aggregate_class_metrics(states)
    assert result["student_count"] == 5
    assert result["engaged_count"] == 5
    assert result["class_engagement"] == pytest.approx(85.0)
    assert result["yawn_rate"] == 0.0


def test_aggregate_mixed():
    states = [
        make_state(0, 0.80, True),
        make_state(1, 0.40, False),
        make_state(2, 0.20, False, yawn_score=0.0),  # yawning
    ]
    result = aggregate_class_metrics(states)
    assert result["student_count"] == 3
    assert result["engaged_count"] == 1
    assert result["yawn_rate"] == pytest.approx(100 / 3, rel=1e-2)
