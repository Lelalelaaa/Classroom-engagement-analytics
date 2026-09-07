"""
Identity tracking for seated students.

Before this existed, `face_id` was the detector's per-frame scan index, so any
change in detection order silently handed one student's cached state to
another. Every downstream feature — per-student calibration, EMA smoothing,
rolling on-task ratio — depends on an identity that survives across frames.

Seated students barely move, so greedy IoU matching with a centroid fallback
and a short lost-track buffer is sufficient; a Kalman/appearance tracker would
add cost and failure modes for motion that does not occur here.
"""
from dataclasses import dataclass, field
from typing import Optional

from .yunet_detector import FaceDetection


@dataclass
class Track:
    track_id: int
    detection: FaceDetection
    lost_frames: int = 0
    age: int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        d = self.detection
        return d.x + d.w / 2.0, d.y + d.h / 2.0


@dataclass
class FaceTracker:
    iou_threshold: float = 0.3
    max_lost_frames: int = 15
    _tracks: dict[int, Track] = field(default_factory=dict)
    _next_id: int = 0

    def update(self, detections: list[FaceDetection]) -> list[tuple[int, FaceDetection]]:
        """
        Assign a stable track_id to each detection in this frame.

        Returns (track_id, detection) pairs in the order given. Tracks not
        matched this frame are retained for `max_lost_frames` so a brief
        detector miss does not reset a student's calibration and history.
        """
        if not detections:
            self._age_unmatched(set())
            return []

        track_ids = list(self._tracks.keys())
        candidates = []
        for ti in track_ids:
            for di, det in enumerate(detections):
                score = self._match_score(self._tracks[ti], det)
                if score > 0.0:
                    candidates.append((score, ti, di))
        candidates.sort(reverse=True)

        assigned_tracks: set[int] = set()
        assigned_dets: dict[int, int] = {}
        for _, ti, di in candidates:
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(ti)
            assigned_dets[di] = ti

        results: list[tuple[int, FaceDetection]] = []
        for di, det in enumerate(detections):
            ti = assigned_dets.get(di)
            if ti is None:
                ti = self._spawn(det)
            else:
                track = self._tracks[ti]
                track.detection = det
                track.lost_frames = 0
                track.age += 1
            results.append((ti, det))

        self._age_unmatched(assigned_tracks)
        return results

    def _spawn(self, det: FaceDetection) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = Track(track_id=track_id, detection=det)
        return track_id

    def _age_unmatched(self, matched: set[int]) -> None:
        for track_id in list(self._tracks):
            if track_id in matched:
                continue
            track = self._tracks[track_id]
            track.lost_frames += 1
            if track.lost_frames > self.max_lost_frames:
                del self._tracks[track_id]

    def _match_score(self, track: Track, det: FaceDetection) -> float:
        iou = _iou(track.detection, det)
        if iou >= self.iou_threshold:
            return iou
        # Fallback for a student who leaned or was briefly missed: accept a
        # detection whose centre sits well inside a face-width of the track.
        # Scored below any real IoU match so overlaps always win.
        tx, ty = track.centroid
        dx, dy = det.x + det.w / 2.0, det.y + det.h / 2.0
        reach = max(track.detection.w, det.w) * 0.6
        if reach <= 0:
            return 0.0
        dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
        if dist < reach:
            return 0.5 * self.iou_threshold * (1.0 - dist / reach)
        return 0.0

    @property
    def active_track_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks.values() if t.lost_frames == 0]

    def get(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)


def _iou(a: FaceDetection, b: FaceDetection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
