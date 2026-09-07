#!/usr/bin/env python3
"""
Refit the attention thresholds against a teacher's segment ratings.

    python eval/tune_from_feedback.py eval/out/lesson1.jsonl segment_ratings.csv \
        --out eval/configs/tuned.yaml

Inputs:

  jsonl   the extract cache `run_eval.py` wrote for that same recording
          (decode + detect + pose + gaze, already done once).
  csv     the ratings the teacher produced in the /review page, or that
          /api/eval/ratings wrote: one row per 5-minute segment, rated
          focused | mixed | off, with a sure/unsure confidence.

For each candidate threshold set we replay the cache, take the class on-task
ratio in each rated segment, and score it against the band that rating implies
(focused >= .75, mixed .40-.75, off < .40). The config with the lowest
confidence-weighted distance to the bands wins and is written to --out.

Coarse by design: `sweep.py` against per-frame labels is the finer tool. This
one exists so a teacher who only watched their lesson back and said "engaged /
so-so / not really" every five minutes can still move the model.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings                                   # noqa: E402
from app.pipeline.engagement_engine import (                      # noqa: E402
    AttentionConfig, AttentionEngine,
)
from app.pipeline.head_pose import HeadPose                       # noqa: E402
from eval.run_eval import Record, _apply_oracle_references        # noqa: E402
from eval.tuning import (                                         # noqa: E402
    SEGMENT_SECONDS, TUNED_FIELDS, band_distance, candidate_configs,
    load_segment_ratings,
)


def segment_ratios(records: list[Record], config: AttentionConfig, calibrate: str,
                   calibration_seconds: float, segment_seconds: float) -> dict[int, float]:
    """Mean class on-task ratio per segment index, replaying the cache once."""
    records = sorted(records, key=lambda r: (r.frame_idx, r.track_id if r.track_id is not None else -1))
    engine = AttentionEngine(config)

    if calibrate == "oracle":
        _apply_oracle_references(engine, records)
        calibration_end = None
    else:
        first_t = records[0].t if records else 0.0
        engine.start_calibration(first_t)
        calibration_end = first_t + calibration_seconds

    buckets: dict[int, list[float]] = {}
    for i, record in enumerate(records):
        frame_end = i + 1 == len(records) or records[i + 1].frame_idx != record.frame_idx

        if record.detected and record.track_id is not None:
            if calibration_end is not None and record.t >= calibration_end:
                engine.finish_calibration()
                calibration_end = None
            engine.observe(
                track_id=record.track_id,
                now=record.t,
                head_pose=None if record.pose is None else HeadPose(*record.pose),
                gaze=None if record.gaze is None else tuple(record.gaze),
                face_px=record.face_px,
            )

        if frame_end:
            metrics = engine.class_metrics(record.t)
            if metrics["tracked_count"]:
                idx = int(record.t // segment_seconds)
                buckets.setdefault(idx, []).append(metrics["on_task_ratio"])

    return {idx: sum(vals) / len(vals) for idx, vals in buckets.items()}


def evaluate(seg_ratios: dict[int, float], ratings, segment_seconds: float) -> dict:
    total_weight = total_error = 0.0
    hits = rated = 0
    rows = []
    for r in ratings:
        idx = int(r.start_s // segment_seconds)
        if idx not in seg_ratios:
            continue
        ratio = seg_ratios[idx]
        dist = band_distance(ratio, r.band)
        total_error += dist * r.weight
        total_weight += r.weight
        rated += 1
        hits += dist == 0.0
        rows.append({
            "segment": f"{int(r.start_s)}-{int(r.end_s)}s",
            "teacher": r.rating,
            "confidence": r.confidence,
            "engine_ratio": round(ratio, 3),
            "band": list(r.band),
            "in_band": dist == 0.0,
        })
    mean_error = total_error / total_weight if total_weight else float("inf")
    return {
        "mean_band_distance": round(mean_error, 4) if total_weight else None,
        "in_band": hits,
        "rated_segments": rated,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="extract cache from run_eval.py")
    parser.add_argument("ratings", help="segment_ratings.csv")
    parser.add_argument("--calibrate", choices=("video-start", "oracle"), default="video-start")
    parser.add_argument("--segment-seconds", type=float, default=SEGMENT_SECONDS)
    parser.add_argument("--out", help="write the winning fields here as YAML")
    parser.add_argument("--report", help="write the full before/after JSON here")
    args = parser.parse_args()

    records = [Record.from_json(line) for line in open(args.jsonl, encoding="utf-8")]
    ratings = load_segment_ratings(args.ratings)
    if not records:
        raise SystemExit(f"empty cache: {args.jsonl}")
    if not ratings:
        raise SystemExit(f"no ratings in {args.ratings}")

    base = AttentionConfig.from_settings(settings)
    base_eval = evaluate(
        segment_ratios(records, base, args.calibrate, base.calibration_seconds, args.segment_seconds),
        ratings, args.segment_seconds,
    )

    best_config, best_eval = base, base_eval
    considered = 1
    for candidate in candidate_configs(base):
        considered += 1
        cand_eval = evaluate(
            segment_ratios(records, candidate, args.calibrate,
                           candidate.calibration_seconds, args.segment_seconds),
            ratings, args.segment_seconds,
        )
        if cand_eval["mean_band_distance"] is None:
            continue
        better = (
            best_eval["mean_band_distance"] is None
            or cand_eval["mean_band_distance"] < best_eval["mean_band_distance"]
            or (cand_eval["mean_band_distance"] == best_eval["mean_band_distance"]
                and cand_eval["in_band"] > best_eval["in_band"])
        )
        if better:
            best_config, best_eval = candidate, cand_eval

    overrides = {f: getattr(best_config, f) for f in TUNED_FIELDS}
    report = {
        "considered_configs": considered,
        "calibration": args.calibrate,
        "baseline": {k: base_eval[k] for k in ("mean_band_distance", "in_band", "rated_segments")},
        "tuned": {k: best_eval[k] for k in ("mean_band_distance", "in_band", "rated_segments")},
        "overrides": overrides,
        "unchanged": best_config == base,
        "segments": best_eval["rows"],
    }

    print(json.dumps(report, indent=2))
    if report["unchanged"]:
        print("\n[tune] no grid config beat the baseline — thresholds left as they are")

    if args.out:
        import yaml
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# Generated by eval/tune_from_feedback.py — teacher segment ratings.\n")
            fh.write(f"# baseline mean band distance {base_eval['mean_band_distance']}"
                     f" -> tuned {best_eval['mean_band_distance']}"
                     f" ({best_eval['in_band']}/{best_eval['rated_segments']} segments in band)\n")
            yaml.safe_dump(overrides, fh, sort_keys=True)
        print(f"[tune] overrides -> {args.out}")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[tune] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
