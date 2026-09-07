"""
Shared pieces for the two threshold fitters:

  sweep.py              fits against per-frame ground-truth labels (fine)
  tune_from_feedback.py fits against a teacher's 5-minute segment ratings (coarse)

Both walk the same small search grid and write the winning fields to a YAML that
`run_eval.load_config` / `AttentionConfig.from_yaml` can load.
"""
import csv
import itertools
from dataclasses import dataclass, replace
from typing import Iterator

from app.pipeline.engagement_engine import AttentionConfig

SEGMENT_SECONDS = 300.0

# rating -> [low, high) target band for the class on-task ratio in that segment.
# Band edges live here, not in the ratings CSV, so a refit can move them without
# re-collecting teacher input.
RATING_BANDS: dict[str, tuple[float, float]] = {
    "focused": (0.75, 1.01),
    "mixed":   (0.40, 0.75),
    "off":     (0.00, 0.40),
}
CONFIDENCE_WEIGHT: dict[str, float] = {"sure": 1.0, "unsure": 0.5}

# Deliberately coarse. A handful of recordings cannot justify a fine fit of 11
# parameters, and a teacher rating is itself a 3-level judgement — the grid is
# meant to move the thresholds in the right direction, not find a global optimum.
SEARCH_GRID: dict[str, list[float]] = {
    "yaw_enter_deg":       [18.0, 22.0, 25.0, 28.0, 32.0],
    "yaw_exit_deg":        [28.0, 32.0, 36.0, 40.0],
    "desk_work_pitch_deg": [12.0, 15.0, 18.0, 22.0],
    "desk_work_weight":    [0.0, 0.5, 0.75, 1.0],
    "on_task_floor":       [0.4, 0.5, 0.6],
}
TUNED_FIELDS: tuple[str, ...] = tuple(SEARCH_GRID)


def candidate_configs(base: AttentionConfig) -> Iterator[AttentionConfig]:
    """Every grid combination, minus the ones that break the enter<exit invariant."""
    keys = list(SEARCH_GRID)
    for combo in itertools.product(*(SEARCH_GRID[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        if overrides["yaw_exit_deg"] <= overrides["yaw_enter_deg"]:
            continue
        yield replace(base, **overrides)


@dataclass(frozen=True)
class SegmentRating:
    session_id: str
    start_s: float
    end_s: float
    rating: str
    confidence: str
    note: str

    @property
    def band(self) -> tuple[float, float]:
        return RATING_BANDS[self.rating]

    @property
    def weight(self) -> float:
        return CONFIDENCE_WEIGHT.get(self.confidence, 1.0)


def load_segment_ratings(path: str) -> list[SegmentRating]:
    """Read the CSV the /review page produces (or /api/eval/ratings writes)."""
    out: list[SegmentRating] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            rating = (row.get("rating") or "").strip().lower()
            if rating not in RATING_BANDS:
                raise ValueError(
                    f"{path}:{line_no}: bad rating {rating!r}; expected one of {sorted(RATING_BANDS)}"
                )
            confidence = (row.get("confidence") or "sure").strip().lower()
            out.append(SegmentRating(
                session_id=(row.get("session_id") or "").strip(),
                start_s=float(row["segment_start_s"]),
                end_s=float(row["segment_end_s"]),
                rating=rating,
                confidence=confidence if confidence in CONFIDENCE_WEIGHT else "sure",
                note=(row.get("note") or "").strip(),
            ))
    return out


def band_distance(ratio: float, band: tuple[float, float]) -> float:
    """0 inside the band, otherwise how far outside it the ratio sits."""
    low, high = band
    if ratio < low:
        return low - ratio
    if ratio > high:
        return ratio - high
    return 0.0
