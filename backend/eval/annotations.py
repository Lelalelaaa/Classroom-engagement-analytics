"""
Ground-truth annotation loading.

CSV schema (one row per visible student per sampled frame):

    frame_idx,student_id,x,y,w,h,label[,row]

  frame_idx   0-based index into the video file
  student_id  stable id for that seat across the whole recording
  x,y,w,h     face bounding box in video pixels
  label       on_board | desk_work | off_board | not_visible
  row         optional seat row (1 = front). Enables the distance breakdown,
              which is the number that says whether the back of the room works.

`not_visible` is a first-class label, not a missing row: it records that an
annotator looked and could not tell, which must not be scored as an error
against the system.
"""
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

VALID_LABELS = {"on_board", "desk_work", "off_board", "not_visible"}

# Ground-truth label -> the AttentionState value the pipeline should emit.
LABEL_TO_STATE = {
    "on_board":  "on_task",
    "desk_work": "desk_work",
    "off_board": "off_task",
}


@dataclass(frozen=True)
class Annotation:
    frame_idx: int
    student_id: str
    x: int
    y: int
    w: int
    h: int
    label: str
    row: Optional[int] = None

    @property
    def is_scorable(self) -> bool:
        return self.label in LABEL_TO_STATE

    @property
    def size_px(self) -> int:
        return min(self.w, self.h)


def load_annotations(path: str) -> dict[int, list[Annotation]]:
    """Return annotations grouped by frame index."""
    by_frame: dict[int, list[Annotation]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            label = (row.get("label") or "").strip()
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_no}: unknown label {label!r}; expected one of {sorted(VALID_LABELS)}"
                )
            seat_row = (row.get("row") or "").strip()
            ann = Annotation(
                frame_idx=int(row["frame_idx"]),
                student_id=row["student_id"].strip(),
                x=int(float(row["x"])), y=int(float(row["y"])),
                w=int(float(row["w"])), h=int(float(row["h"])),
                label=label,
                row=int(seat_row) if seat_row else None,
            )
            by_frame[ann.frame_idx].append(ann)
    return dict(by_frame)


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Inter-annotator agreement on the overlap subset.

    Reported alongside the accuracy numbers because a system cannot
    meaningfully be judged more consistent than the humans it is measured
    against — a low kappa caps how much any F1 here can be trusted.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa needs two equal-length, non-empty label lists")
    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
