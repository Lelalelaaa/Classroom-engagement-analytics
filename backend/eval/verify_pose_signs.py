#!/usr/bin/env python3
"""
Sanity-check the head-pose sign convention before any tuning.

    python eval/verify_pose_signs.py eval/out/lesson1.jsonl

`desk_work` classification assumes MediaPipe's transform matrix yields
"positive pitch = looking down" (head_pose.pose_from_matrix, marked
TODO: assumed). If that sign is flipped, every desk_work / off_task call
inverts and a fit on top of it is meaningless.

Checks, over detections the extract stage matched to a human label:

  1. median pitch on `desk_work` frames is clearly greater (more positive)
     than on `on_board` frames — students looking at notes really are pitched
     down relative to students looking at the board.
  2. `off_board` yaw spreads wider than `on_board` yaw — turning away shows up
     as larger yaw magnitude.

Prints the medians and a PASS / FAIL. Non-zero exit on FAIL.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import Record                                  # noqa: E402

PITCH_MARGIN_DEG = 4.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="extract cache from run_eval.py")
    args = parser.parse_args()

    pitch: dict[str, list[float]] = {"on_board": [], "desk_work": []}
    yaw_abs: dict[str, list[float]] = {"on_board": [], "off_board": []}
    for line in open(args.jsonl, encoding="utf-8"):
        record = Record.from_json(line)
        if not record.detected or record.pose is None or record.truth is None:
            continue
        yaw, p = record.pose[0], record.pose[1]
        if record.truth in pitch:
            pitch[record.truth].append(p)
        if record.truth in yaw_abs:
            yaw_abs[record.truth].append(abs(yaw))

    if len(pitch["desk_work"]) < 5 or len(pitch["on_board"]) < 5:
        raise SystemExit("not enough on_board / desk_work samples in the cache to check")

    med_desk = statistics.median(pitch["desk_work"])
    med_board = statistics.median(pitch["on_board"])
    pitch_ok = med_desk - med_board >= PITCH_MARGIN_DEG

    yaw_ok = True
    if len(yaw_abs["off_board"]) >= 5:
        yaw_ok = statistics.median(yaw_abs["off_board"]) > statistics.median(yaw_abs["on_board"])

    print(f"pitch  on_board median={med_board:+.1f}deg  desk_work median={med_desk:+.1f}deg"
          f"  (desk_work should be >= on_board + {PITCH_MARGIN_DEG})")
    if len(yaw_abs["off_board"]) >= 5:
        print(f"|yaw|  on_board median={statistics.median(yaw_abs['on_board']):.1f}deg"
              f"  off_board median={statistics.median(yaw_abs['off_board']):.1f}deg")
    else:
        print("|yaw|  not enough off_board samples — skipped")

    if pitch_ok and yaw_ok:
        print("PASS — sign convention looks right")
        return 0
    print("FAIL — head-pose signs may be flipped; do not tune on this cache")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
