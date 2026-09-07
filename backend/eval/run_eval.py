#!/usr/bin/env python3
"""
Offline evaluation: how accurate is the attention model on real footage?

    python eval/run_eval.py recordings/lesson1.mp4 annotations/lesson1.csv \
        --config eval/configs/baseline.yaml --out eval/out/lesson1.json

Runs in two stages so that threshold sweeps do not re-decode the video:

  extract  decode frames, detect faces, track them, measure head pose and eye
           gaze, and match each detection to its annotation. Written to a
           JSONL cache. This is the expensive stage and depends only on the
           detector/pose settings.
  score    replay the cached measurements through AttentionEngine with a given
           threshold set and report accuracy. Cheap, so a sweep is many score
           passes over one extraction.

Detection recall is reported separately from classification accuracy: an
undetected student is dropped from the metric rather than counted as
disengaged, so poor recall inflates the engagement score instead of lowering
it, and the two failures must not be averaged together.
"""
import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings                                    # noqa: E402
from app.pipeline.engagement_engine import (                       # noqa: E402
    AttentionConfig, AttentionEngine,
)
from app.pipeline.head_pose import HeadPose, pose_from_matrix      # noqa: E402
from app.pipeline.mediapipe_init import FacePipelineConfig         # noqa: E402
from app.pipeline.tracker import FaceTracker                       # noqa: E402
from app.pipeline.yunet_detector import YuNetDetector              # noqa: E402
from eval.annotations import LABEL_TO_STATE, load_annotations      # noqa: E402
from eval.metrics import Scorer, iou                               # noqa: E402

MATCH_IOU = 0.3


@dataclass
class Record:
    frame_idx: int
    t: float
    track_id: Optional[int]
    bbox: Optional[list[int]]
    face_px: int
    pose: Optional[list[float]]
    gaze: Optional[list[float]]
    truth: Optional[str]
    student_id: Optional[str]
    row: Optional[int]
    detected: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(line: str) -> "Record":
        return Record(**json.loads(line))


# ── Stage 1: extraction ───────────────────────────────────────────────────────

def extract(video_path: str, annotations_path: str, cache_path: str, target_fps: float) -> None:
    by_frame = load_annotations(annotations_path)
    annotated = set(by_frame)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / target_fps)))

    detector = YuNetDetector(
        settings.YUNET_MODEL_PATH,
        score_threshold=settings.YUNET_SCORE_THRESHOLD,
        nms_threshold=settings.YUNET_NMS_THRESHOLD,
        tiling=settings.DETECT_TILING,
        tile_overlap=settings.DETECT_TILE_OVERLAP,
    )
    mp_pipeline = FacePipelineConfig(max_faces=settings.MAX_FACES_PER_CAMERA, crop_pool_size=1)
    gaze_model = _load_gaze_model()
    tracker = FaceTracker(
        iou_threshold=settings.TRACK_IOU_THRESHOLD,
        max_lost_frames=settings.TRACK_MAX_LOST_FRAMES,
    )

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    written = frame_idx = 0

    with open(cache_path, "w", encoding="utf-8") as out:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % step and frame_idx not in annotated:
                frame_idx += 1
                continue

            t = frame_idx / video_fps
            detections = detector.detect(frame)
            tracked = tracker.update(detections)
            truths = list(by_frame.get(frame_idx, []))
            claimed: set[int] = set()

            for track_id, det in tracked:
                # Eye gaze is measured for every eligible face here, not
                # round-robin as in the live path: the budget affects refresh
                # latency, not per-sample accuracy, and we want the cleanest
                # measurement of the model itself.
                gaze = None
                if gaze_model is not None and det.size_px >= settings.MIN_FACE_PX_FOR_GAZE:
                    result = gaze_model.estimate(frame, det)
                    if result is not None:
                        gaze = [result.yaw, result.pitch]

                pose = _head_pose(mp_pipeline, frame, det)
                match_i, match = _best_match(det, truths, claimed)
                if match_i is not None:
                    claimed.add(match_i)

                out.write(Record(
                    frame_idx=frame_idx, t=t, track_id=track_id,
                    bbox=[det.x, det.y, det.w, det.h], face_px=det.size_px,
                    pose=None if pose is None else [pose.yaw, pose.pitch, pose.roll],
                    gaze=gaze,
                    truth=match.label if match else None,
                    student_id=match.student_id if match else None,
                    row=match.row if match else None,
                    detected=True,
                ).to_json() + "\n")
                written += 1

            # Annotated students no detection matched — the recall failures.
            for i, ann in enumerate(truths):
                if i in claimed:
                    continue
                out.write(Record(
                    frame_idx=frame_idx, t=t, track_id=None,
                    bbox=[ann.x, ann.y, ann.w, ann.h], face_px=ann.size_px,
                    pose=None, gaze=None, truth=ann.label,
                    student_id=ann.student_id, row=ann.row, detected=False,
                ).to_json() + "\n")
                written += 1

            frame_idx += 1

    capture.release()
    print(f"[extract] {written} records over {len(annotated)} annotated frames -> {cache_path}")


def _load_gaze_model():
    try:
        from app.pipeline.gaze_onnx import GazeEstimatorONNX
        return GazeEstimatorONNX(
            backbone=settings.GAZE_BACKBONE,
            margin=settings.GAZE_CROP_MARGIN,
            intra_op_threads=settings.ONNX_INTRA_OP_THREADS,
        )
    except Exception as exc:
        print(f"[extract] eye gaze unavailable ({exc}) — head pose only")
        return None


def _head_pose(mp_pipeline, frame: np.ndarray, det) -> Optional[HeadPose]:
    h, w = frame.shape[:2]
    mx, my = int(det.w * 0.15), int(det.h * 0.15)
    x1, y1 = max(0, det.x - mx), max(0, det.y - my)
    x2, y2 = min(w, det.x + det.w + mx), min(h, det.y + det.h + my)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    result = mp_pipeline.process_crop(frame[y1:y2, x1:x2])
    if not result.facial_transformation_matrixes:
        return None
    return pose_from_matrix(result.facial_transformation_matrixes[0])


def _best_match(det, truths, claimed):
    best_i, best_iou = None, MATCH_IOU
    for i, ann in enumerate(truths):
        if i in claimed:
            continue
        score = iou((det.x, det.y, det.w, det.h), (ann.x, ann.y, ann.w, ann.h))
        if score >= best_iou:
            best_i, best_iou = i, score
    return best_i, (truths[best_i] if best_i is not None else None)


# ── Stage 2: scoring ──────────────────────────────────────────────────────────

def score(cache_path: str, config: AttentionConfig, calibrate: str,
          calibration_seconds: float) -> dict:
    records = [Record.from_json(line) for line in open(cache_path, encoding="utf-8")]
    return score_records(records, config, calibrate, calibration_seconds)


def score_records(records: list["Record"], config: AttentionConfig, calibrate: str,
                  calibration_seconds: float) -> dict:
    """`score` on already-loaded records, so a sweep replays one parsed cache."""
    records = sorted(records, key=lambda r: (r.frame_idx, r.track_id if r.track_id is not None else -1))

    engine = AttentionEngine(config)
    scorer = Scorer()

    if calibrate == "oracle":
        _apply_oracle_references(engine, records)
        calibration_end = None
    else:
        first_t = records[0].t if records else 0.0
        engine.start_calibration(first_t)
        calibration_end = first_t + calibration_seconds

    for record in records:
        if not record.detected or record.track_id is None:
            if record.truth in LABEL_TO_STATE:
                scorer.add(LABEL_TO_STATE[record.truth], None, record.face_px, record.row, False)
            continue

        if calibration_end is not None and record.t >= calibration_end:
            engine.finish_calibration()
            calibration_end = None

        student = engine.observe(
            track_id=record.track_id,
            now=record.t,
            head_pose=None if record.pose is None else HeadPose(*record.pose),
            gaze=None if record.gaze is None else tuple(record.gaze),
            face_px=record.face_px,
        )
        if record.truth in LABEL_TO_STATE:
            scorer.add(
                LABEL_TO_STATE[record.truth], student.state.value,
                record.face_px, record.row, True,
            )

    report = scorer.report()
    report["calibration"] = calibrate
    report["config"] = config.__dict__
    return report


def _apply_oracle_references(engine: AttentionEngine, records: list[Record]) -> None:
    """
    Best-case calibration: each track's reference is the median pose over the
    frames a human labelled `on_board`.

    This measures the ceiling the calibration step is aiming at. It is
    optimistic by construction — it uses the answer key — so it is reported as
    a separate mode and must never be quoted as the system's accuracy.
    """
    from app.pipeline.engagement_engine import StudentAttention

    poses: dict[int, list[tuple[float, float]]] = {}
    for r in records:
        if r.detected and r.pose is not None and r.truth == "on_board" and r.track_id is not None:
            poses.setdefault(r.track_id, []).append((r.pose[0], r.pose[1]))

    for track_id, samples in poses.items():
        if len(samples) < 3:
            continue
        student = engine.students.setdefault(track_id, StudentAttention(track_id=track_id))
        student.reference = HeadPose(
            yaw=statistics.median([y for y, _ in samples]),
            pitch=statistics.median([p for _, p in samples]),
            roll=0.0,
        )


def load_config(path: Optional[str]) -> AttentionConfig:
    base = AttentionConfig.from_settings(settings)
    if not path:
        return base
    import yaml
    with open(path, encoding="utf-8") as fh:
        overrides = yaml.safe_load(fh) or {}
    unknown = set(overrides) - set(base.__dict__)
    if unknown:
        raise SystemExit(f"Unknown config keys in {path}: {sorted(unknown)}")
    return AttentionConfig(**{**base.__dict__, **overrides})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("annotations")
    parser.add_argument("--config", help="YAML overriding AttentionConfig fields")
    parser.add_argument("--cache", help="extraction cache path (default: eval/out/<video>.jsonl)")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="skip extraction if the cache already exists")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="frames per second to process (default: the live pipeline's 2)")
    parser.add_argument("--calibrate", choices=("video-start", "oracle"), default="video-start")
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    cache = args.cache or os.path.join(os.path.dirname(__file__), "out", f"{stem}.jsonl")

    if not (args.reuse_cache and os.path.exists(cache)):
        extract(args.video, args.annotations, cache, args.fps)

    config = load_config(args.config)
    report = score(cache, config, args.calibrate, config.calibration_seconds)

    text = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[score] report -> {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
