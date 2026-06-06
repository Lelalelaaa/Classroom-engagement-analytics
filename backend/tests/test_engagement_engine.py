import pytest
from app.pipeline.gaze_estimator import GazeResult
from app.pipeline.yawn_detector import YawnResult
from app.pipeline.emotion_classifier import EmotionResult
from app.pipeline.engagement_engine import compute_engagement, WEIGHTS, EMOTION_VALENCE


def make_results(looking: bool, emotion: str, yawning: bool):
    return (
        GazeResult(face_id=0, is_looking_at_screen=looking, gaze_ratio=0.5),
        EmotionResult(dominant_emotion=emotion),
        YawnResult(is_yawning=yawning, mar_score=0.5),
    )


def test_fully_engaged():
    g, em, y = make_results(looking=True, emotion="happy", yawning=False)
    state    = compute_engagement(0, g, em, y)
    assert state.composite_score >= 0.9
    assert state.is_engaged


def test_fully_disengaged():
    g, em, y = make_results(looking=False, emotion="sad", yawning=True)
    state    = compute_engagement(0, g, em, y)
    assert state.composite_score <= 0.3
    assert not state.is_engaged


def test_score_formula_weights():
    """Verify the exact formula: E = 0.50*G + 0.30*Em + 0.20*Y"""
    g, em, y = make_results(looking=True, emotion="neutral", yawning=False)
    state    = compute_engagement(0, g, em, y)
    expected = (
        WEIGHTS["gaze"]    * 1.0 +
        WEIGHTS["emotion"] * EMOTION_VALENCE["neutral"] +
        WEIGHTS["yawn"]    * 1.0
    )
    assert abs(state.composite_score - expected) < 0.001


def test_yawning_reduces_score():
    g1, em1, y1 = make_results(looking=True, emotion="happy", yawning=False)
    g2, em2, y2 = make_results(looking=True, emotion="happy", yawning=True)
    score_no_yawn = compute_engagement(0, g1, em1, y1).composite_score
    score_yawning = compute_engagement(0, g2, em2, y2).composite_score
    assert score_no_yawn > score_yawning
