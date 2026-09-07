"""Scoring for the offline evaluation: detection recall, per-class P/R/F1, breakdowns."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

ATTENTIVE_STATES = {"on_task", "desk_work"}


@dataclass
class Case:
    """One annotated student in one frame, with whatever the pipeline said."""
    truth: str
    predicted: Optional[str]
    size_px: int
    row: Optional[int]
    detected: bool


@dataclass
class Scorer:
    cases: list[Case] = field(default_factory=list)

    def add(self, truth: str, predicted: Optional[str], size_px: int,
            row: Optional[int], detected: bool) -> None:
        self.cases.append(Case(truth, predicted, size_px, row, detected))

    def report(self) -> dict:
        return {
            "n_annotated":      len(self.cases),
            "detection_recall": self._detection_recall(self.cases),
            "overall":          self._classification(self.cases),
            "by_row":           self._grouped(lambda c: c.row, "row"),
            "by_face_px":       self._grouped(lambda c: _size_bucket(c.size_px), "face_px"),
        }

    def _grouped(self, key, label: str) -> dict:
        groups: dict = defaultdict(list)
        for case in self.cases:
            groups[key(case)].append(case)
        return {
            str(k): {
                "n": len(v),
                "detection_recall": self._detection_recall(v),
                **self._classification(v),
            }
            for k, v in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0]))
        }

    @staticmethod
    def _detection_recall(cases: list[Case]) -> Optional[float]:
        """
        Share of annotated students the detector found at all.

        Tracked separately from classification because an undetected student is
        a different failure: they vanish from the denominator instead of
        lowering the score, so poor recall inflates engagement rather than
        depressing it.
        """
        if not cases:
            return None
        return round(sum(1 for c in cases if c.detected) / len(cases), 4)

    @staticmethod
    def _classification(cases: list[Case]) -> dict:
        scored = [c for c in cases if c.predicted is not None and c.predicted != "unknown"]
        if not scored:
            return {"n_scored": 0, "binary_f1": None, "accuracy": None, "confusion": {}}

        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in scored:
            confusion[c.truth][c.predicted] += 1

        correct = sum(1 for c in scored if c.truth == c.predicted)

        # Binary view: attentive (on-task or desk-work) vs off-task. This is
        # the claim the dashboard actually makes, so it is the headline number.
        tp = sum(1 for c in scored if c.truth in ATTENTIVE_STATES and c.predicted in ATTENTIVE_STATES)
        fp = sum(1 for c in scored if c.truth not in ATTENTIVE_STATES and c.predicted in ATTENTIVE_STATES)
        fn = sum(1 for c in scored if c.truth in ATTENTIVE_STATES and c.predicted not in ATTENTIVE_STATES)

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else None
        )

        return {
            "n_scored":  len(scored),
            "accuracy":  round(correct / len(scored), 4),
            "precision": None if precision is None else round(precision, 4),
            "recall":    None if recall is None else round(recall, 4),
            "binary_f1": None if f1 is None else round(f1, 4),
            "confusion": {t: dict(p) for t, p in confusion.items()},
        }


def _size_bucket(px: int) -> str:
    for edge in (24, 40, 60, 80, 120):
        if px < edge:
            return f"<{edge}"
    return ">=120"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
