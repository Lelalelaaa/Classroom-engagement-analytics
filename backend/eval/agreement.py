#!/usr/bin/env python3
"""
Inter-annotator agreement between two annotation files.

    python eval/agreement.py annotations/lesson1_a.csv annotations/lesson1_b.csv

Compares the rows both annotators labelled (matched on frame_idx + student_id)
and reports Cohen's kappa plus the disagreements.

Why this is reported alongside accuracy: the system is scored against these
labels, so their consistency bounds how far any F1 from `run_eval.py` can be
trusted. A kappa of 0.5 means the humans agreed little more than chance would
predict, and an F1 measured against them means correspondingly little.
"""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.annotations import cohens_kappa  # noqa: E402


def _load(path: str) -> dict[tuple[int, str], str]:
    labels: dict[tuple[int, str], str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            student = (row.get("student_id") or "").strip()
            label = (row.get("label") or "").strip()
            if student and label:
                labels[(int(row["frame_idx"]), student)] = label
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--show", type=int, default=15, help="how many disagreements to list")
    args = parser.parse_args()

    a, b = _load(args.file_a), _load(args.file_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(
            "No overlapping (frame_idx, student_id) rows. The second annotator must "
            "label the SAME frames and use the SAME student ids."
        )

    labels_a = [a[k] for k in shared]
    labels_b = [b[k] for k in shared]
    kappa = cohens_kappa(labels_a, labels_b)
    agreed = sum(1 for x, y in zip(labels_a, labels_b) if x == y)

    print(f"overlap:      {len(shared)} labelled rows")
    print(f"raw agreement {agreed / len(shared):.1%}")
    print(f"Cohen's kappa {kappa:.3f}   ({_interpret(kappa)})")

    disagreements = [(k, a[k], b[k]) for k in shared if a[k] != b[k]]
    if disagreements:
        pairs = Counter(tuple(sorted((x, y))) for _, x, y in disagreements)
        print("\nmost common confusions:")
        for (x, y), n in pairs.most_common(5):
            print(f"  {x:<12} vs {y:<12} {n}")
        print(f"\nfirst {min(args.show, len(disagreements))} disagreements (frame, student, A, B):")
        for (frame, student), x, y in disagreements[: args.show]:
            print(f"  {frame:>7}  {student:<6} {x:<12} {y}")
    return 0


def _interpret(kappa: float) -> str:
    """Landis & Koch (1977) bands — the convention these numbers are read against."""
    for threshold, wording in ((0.81, "almost perfect"), (0.61, "substantial"),
                               (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if kappa >= threshold:
            return wording
    return "poor — worse than chance"


if __name__ == "__main__":
    raise SystemExit(main())
