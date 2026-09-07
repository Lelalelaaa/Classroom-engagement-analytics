import csv

import pytest

from app.pipeline.engagement_engine import AttentionConfig
from eval.tuning import (
    RATING_BANDS, band_distance, candidate_configs, load_segment_ratings,
)


def test_band_distance_is_zero_inside_and_positive_outside():
    assert band_distance(0.80, RATING_BANDS["focused"]) == 0.0
    assert band_distance(0.50, RATING_BANDS["mixed"]) == 0.0
    assert band_distance(0.10, RATING_BANDS["off"]) == 0.0

    assert band_distance(0.60, RATING_BANDS["focused"]) == pytest.approx(0.15)
    assert band_distance(0.90, RATING_BANDS["off"]) == pytest.approx(0.50)


def test_candidate_grid_keeps_exit_above_enter():
    base = AttentionConfig()
    configs = list(candidate_configs(base))
    assert configs, "grid produced no candidates"
    assert all(c.yaw_exit_deg > c.yaw_enter_deg for c in configs)
    # every candidate stays a valid AttentionConfig with only grid fields moved
    for c in configs:
        assert c.ema_alpha == base.ema_alpha
        assert c.calibration_seconds == base.calibration_seconds


def test_load_segment_ratings_parses_and_defaults_confidence(tmp_path):
    path = tmp_path / "segment_ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "segment_start_s", "segment_end_s", "rating", "confidence", "note"])
        writer.writerow(["5b-0917", 0, 300, "focused", "sure", ""])
        writer.writerow(["5b-0917", 300, 600, "mixed", "", "group work"])
        writer.writerow(["5b-0917", 600, 900, "off", "unsure", ""])

    ratings = load_segment_ratings(str(path))
    assert [r.rating for r in ratings] == ["focused", "mixed", "off"]
    assert ratings[1].confidence == "sure"      # blank -> sure
    assert ratings[1].weight == 1.0
    assert ratings[2].weight == 0.5             # unsure is half weight
    assert ratings[0].band == RATING_BANDS["focused"]


def test_load_segment_ratings_rejects_unknown_rating(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "segment_start_s", "segment_end_s", "rating", "confidence", "note"])
        writer.writerow(["x", 0, 300, "distracted", "sure", ""])

    with pytest.raises(ValueError):
        load_segment_ratings(str(path))
