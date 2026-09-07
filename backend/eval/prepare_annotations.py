#!/usr/bin/env python3
"""
Turn a recorded lesson into an annotation job.

    python eval/prepare_annotations.py recordings/lesson1.mp4 \
        --every 5 --out annotations/lesson1

Samples one frame every N seconds, runs the detector on it, and writes:

    <out>/frames/frame_000450.jpg   the sampled frames, boxes drawn and numbered
    <out>/lesson1.csv               one pre-filled row per detected face
    <out>/README.md                 what the annotator has to do

Pre-filling the boxes matters: hand-drawing bounding boxes for ~20 students
across ~300 frames is the part that makes ground truth expensive, and it is
also the part a detector can do. The annotator supplies the labels, which is
the part it cannot.

The detector will miss students. Those misses are the detection-recall number
we most need, so the protocol asks the annotator to ADD a row for any student
the boxes missed rather than ignoring them.
"""
import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings                          # noqa: E402
from app.pipeline.yunet_detector import YuNetDetector    # noqa: E402

README = """# Annotation job: {stem}

{n_frames} frames sampled every {every}s from `{video}`.

## What to do

Open `{stem}.csv` in a spreadsheet. Each row is one detected face in one
frame. The `student_id` and `label` columns are empty — fill both in.

Look at `frames/frame_XXXXXX.jpg`; every box is numbered with its `box_idx`.

### student_id
Give each SEAT a stable id (`s1`, `s2`, …) and use the same id for that person
in every frame. This is what lets accuracy be broken down per student and per
row. Keep a seat map next to you.

### label — one of exactly these four

| label | means |
|---|---|
| `on_board` | looking at the board, the teacher, or the projected slides |
| `desk_work` | looking down at notes, a book, or a laptop — still on task |
| `off_board` | looking away: at a neighbour, out the window, at a phone |
| `not_visible` | you cannot tell — occluded, blurred, turned away, out of frame |

`not_visible` is a real answer, not a failure. Use it whenever you would be
guessing. A forced guess is worse than an honest gap, because the system is
scored against these labels.

### row
Seat row, 1 = closest to the camera. This produces the accuracy-by-distance
breakdown, which is the number that says whether the back of the room works.

## Students the detector missed

If a student is clearly visible but has no box, ADD a row for them: fill in
`frame_idx`, `student_id`, `x,y,w,h` (approximate is fine), `label` and `row`,
and leave `box_idx` empty. These rows are how detection recall gets measured,
so please do not skip them.

## Second annotator

Have a second person independently label at least 10% of the frames into a
separate file. Comparing the two gives Cohen's kappa, which bounds how far any
accuracy figure here can be trusted — the system cannot meaningfully be judged
more consistent than the people it is measured against.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between sampled frames")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--max-frames", type=int, default=400)
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * args.every)))

    detector = YuNetDetector(
        settings.YUNET_MODEL_PATH,
        score_threshold=settings.YUNET_SCORE_THRESHOLD,
        nms_threshold=settings.YUNET_NMS_THRESHOLD,
        tiling=settings.DETECT_TILING,
        tile_overlap=settings.DETECT_TILE_OVERLAP,
    )

    csv_path = os.path.join(args.out, f"{stem}.csv")
    frame_idx = sampled = total_faces = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "student_id", "x", "y", "w", "h", "label", "row", "box_idx"])

        while sampled < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % step:
                frame_idx += 1
                continue

            detections = sorted(detector.detect(frame), key=lambda d: (d.y, d.x))
            annotated = frame.copy()
            for box_idx, det in enumerate(detections):
                writer.writerow([frame_idx, "", det.x, det.y, det.w, det.h, "", "", box_idx])
                cv2.rectangle(annotated, (det.x, det.y),
                              (det.x + det.w, det.y + det.h), (0, 220, 0), 2)
                cv2.putText(annotated, str(box_idx), (det.x, max(12, det.y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg"), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            total_faces += len(detections)
            sampled += 1
            frame_idx += 1

    capture.release()

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(stem=stem, n_frames=sampled, every=args.every, video=args.video))

    print(f"[prepare] {sampled} frames, {total_faces} pre-filled boxes "
          f"({total_faces / sampled:.1f} per frame)" if sampled else "[prepare] no frames sampled")
    print(f"[prepare] annotator starts at {os.path.join(args.out, 'README.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
