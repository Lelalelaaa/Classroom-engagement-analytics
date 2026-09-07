"""
Collects a teacher's segment ratings for offline threshold tuning.

Writes `eval/ratings/<session_id>.csv` in the schema `eval/tuning.load_segment_ratings`
reads, so `eval/tune_from_feedback.py` consumes it directly. No DB row: the eval
pipeline is file-in / file-out. This is experiment tooling for consented,
already-recorded lessons — it is not part of the live privacy-bounded path and
stores no video and no per-student data.
"""
import csv
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_RATINGS = {"focused", "mixed", "off"}
_CONFIDENCE = {"sure", "unsure"}
_RATINGS_DIR = Path(__file__).resolve().parents[3] / "eval" / "ratings"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


class SegmentRatingIn(BaseModel):
    segment_start_s: float = Field(ge=0)
    segment_end_s: float = Field(gt=0)
    rating: str
    confidence: str = "sure"
    note: str = ""


class RatingsSubmission(BaseModel):
    session_id: str
    segments: list[SegmentRatingIn]


@router.post("/ratings", status_code=201)
async def submit_ratings(payload: RatingsSubmission):
    if not payload.segments:
        raise HTTPException(status_code=422, detail="no segments")
    for seg in payload.segments:
        if seg.rating not in _RATINGS:
            raise HTTPException(status_code=422, detail=f"bad rating {seg.rating!r}")
        if seg.confidence not in _CONFIDENCE:
            raise HTTPException(status_code=422, detail=f"bad confidence {seg.confidence!r}")

    safe_id = _UNSAFE.sub("-", payload.session_id).strip("-") or "unnamed"
    _RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RATINGS_DIR / f"{safe_id}.csv"

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "segment_start_s", "segment_end_s", "rating", "confidence", "note"])
        for seg in sorted(payload.segments, key=lambda s: s.segment_start_s):
            writer.writerow([
                safe_id, int(seg.segment_start_s), int(seg.segment_end_s),
                seg.rating, seg.confidence, seg.note.replace("\n", " ").strip(),
            ])

    return {"saved": f"eval/ratings/{safe_id}.csv", "segments": len(payload.segments)}
