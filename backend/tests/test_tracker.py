from app.pipeline.tracker import FaceTracker
from app.pipeline.yunet_detector import FaceDetection


def _det(x, y, size=100):
    return FaceDetection(x=x, y=y, w=size, h=size, score=0.9)


def test_identity_survives_detection_reordering():
    """
    The regression this exists for: face_id used to be the detector's scan
    index, so a reordered frame handed one student's state to another.
    """
    tracker = FaceTracker()
    tracker.update([_det(100, 100), _det(400, 100)])
    # same two students, jittered and reported in the opposite order
    reordered = {det.x: track_id for track_id, det in tracker.update([_det(404, 103), _det(97, 102)])}
    assert reordered[404] == 1
    assert reordered[97] == 0


def test_track_survives_a_brief_detector_miss():
    tracker = FaceTracker(max_lost_frames=3)
    tracker.update([_det(100, 100), _det(400, 100)])
    tracker.update([_det(100, 100)])
    reappeared = tracker.update([_det(400, 100)])
    assert reappeared[0][0] == 1


def test_track_is_recycled_after_the_lost_buffer():
    tracker = FaceTracker(max_lost_frames=2)
    tracker.update([_det(100, 100), _det(400, 100)])
    for _ in range(4):
        tracker.update([_det(100, 100)])
    assert tracker.update([_det(400, 100)])[0][0] == 2


def test_active_ids_exclude_lost_tracks():
    tracker = FaceTracker(max_lost_frames=5)
    tracker.update([_det(100, 100), _det(400, 100)])
    tracker.update([_det(100, 100)])
    assert tracker.active_track_ids == [0]
