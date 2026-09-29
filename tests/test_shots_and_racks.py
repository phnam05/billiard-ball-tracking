"""What changed on 29 Sep 2026: a ball frozen on the far cushion, two balls
that cannot overlap, a potted number held for its rack, and shots on real
footage (the answer keys' shot lists)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from billiards import Config, ballnet
from billiards import balls as ballnum
from billiards.detect import ColorSignature, Detection
from billiards.events import EventDetector, EventType
from billiards.geometry import TableModel
from billiards.shots import ShotSegmenter
from billiards.track import MultiObjectTracker, TrackSample, TrackState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import real_eval  # noqa: E402


def _table() -> TableModel:
    corners = np.array([[300.0, 180.0], [980.0, 180.0], [1180.0, 600.0], [100.0, 600.0]])
    return TableModel(corners, length_in=100.0, width_in=50.0, ball_diameter_in=2.25)


def _family(name: str, p: float = 0.95) -> np.ndarray:
    fam = ballnum.MODEL_FAMILIES
    v = np.full(len(fam), (1.0 - p) / (len(fam) - 1))
    v[fam.index(name)] = p
    return v


def _det(xy, lab=(200.0, 125.0, 170.0), ball_p=1.0, family="yellow", stripe=0.05,
         band=False, modelled=True) -> Detection:
    return Detection(
        centre_image=(0.0, 0.0), centre_table=tuple(xy), radius_px=10.0, area_ratio=1.0,
        circularity=0.9, signature=ColorSignature(lab=np.array(lab, dtype=np.float64)),
        rim_contrast=40.0, in_raised_band=band, ball_p=ball_p,
        cue_p=0.01 if modelled else None, family_p=_family(family) if modelled else None,
        stripe_p=stripe if modelled else None,
    )


# --------------------------------------------------------------------------
# The far cushion
# --------------------------------------------------------------------------


def test_a_ball_the_model_is_sure_of_can_start_a_track_past_the_far_edge():
    """The tripod answer key's 1 sat against the far cushion from the first
    frame, never on the plain bed, and was missed on 12 of 16 keyframes."""
    tracker = MultiObjectTracker(Config(), _table(), fps=30.0)
    for i in range(6):
        tracker.update([_det((50.0, 49.0), band=True, ball_p=0.9999)], 1 / 30, i, i / 30.0)
    assert [t.state for t in tracker.tracks] == [TrackState.CONFIRMED]


def test_anything_else_past_the_far_edge_still_starts_nothing():
    """A hand on the far rail is there too.  Without the model's word, or with
    it unsure, a detection there follows a ball but starts none."""
    for det in (_det((50.0, 49.0), band=True, ball_p=0.9), _det((50.0, 49.0), band=True, modelled=False)):
        tracker = MultiObjectTracker(Config(), _table(), fps=30.0)
        for i in range(6):
            tracker.update([det], 1 / 30, i, i / 30.0)
        assert tracker.tracks == []


# --------------------------------------------------------------------------
# Two balls cannot overlap
# --------------------------------------------------------------------------


def test_a_detection_on_a_ball_at_rest_is_that_ball_whatever_its_colour():
    """The 1's split-off half, next to the cue ball that rolled up to it,
    sampled the cue ball and the shadow and started a second track on it."""
    cfg = Config()
    tracker = MultiObjectTracker(cfg, _table(), fps=30.0)
    for i in range(10):
        tracker.update([_det((50.0, 25.0))], 1 / 30, i, i / 30.0)
    ball = tracker.tracks[0]
    odd = (60.0, 170.0, 60.0)  # far outside the colour gate
    tracker.update([_det((50.7, 25.0), lab=odd)], 1 / 30, 10, 10 / 30.0)
    assert len(tracker.tracks) == 1 and ball.time_since_update == 0
    # ...and its colour is not learned from such a sighting.
    assert ball.signature.lab[1] < 140.0


# --------------------------------------------------------------------------
# A potted ball's number, for the rest of its rack
# --------------------------------------------------------------------------


def _numbered_tracker(numbers: str, balls):
    cfg = Config()
    cfg.balls.numbers = numbers
    cfg.balls.ball_set = "standard"
    tracker = MultiObjectTracker(cfg, _table(), fps=30.0)
    for i in range(12):
        tracker.update([_det(xy, family=fam, stripe=st) for xy, fam, st in balls], 1 / 30, i, i / 30.0)
    return tracker


def _pot(tracker, track, frame):
    tracker.tracks.remove(track)
    track.kill(frame, "potted")
    tracker.finished.append(track)


def test_a_potted_number_is_not_given_to_a_look_alike_in_the_same_rack():
    """The tripod key's 9 was potted; its far yellow 1, which the ball model
    reads as a stripe, then took the 9's number."""
    balls = [((20.0, 10.0), "yellow", 0.9), ((80.0, 40.0), "blue", 0.05), ((50.0, 25.0), "red", 0.05)]
    tracker = _numbered_tracker("1-15", balls)
    nine = next(t for t in tracker.tracks if t.number == 9)
    _pot(tracker, nine, 12)
    rest = [(xy, fam, st) for xy, fam, st in balls if fam != "yellow"]
    for i in range(13, 25):
        dets = [_det((22.0, 12.0), family="yellow", stripe=0.9)] + [_det(xy, family=f, stripe=st) for xy, f, st in rest]
        tracker.update(dets, 1 / 30, i, i / 30.0)
    look_alike = next(t for t in tracker.tracks if t is not nine and np.allclose(t.kf.position, (22.0, 12.0), atol=1.0))
    assert look_alike.state is TrackState.CONFIRMED and look_alike.clean_samples >= 4
    assert look_alike.number != 9


def test_a_new_rack_frees_the_potted_numbers():
    """More balls on the table than can be left of the rack: a new one.
    The 2026 US Open highlights run two racks."""
    colours = ["yellow", "blue", "red", "purple", "orange", "green", "maroon"]
    first = [((10.0 + 10 * k, 10.0), c, 0.05) for k, c in enumerate(colours)]
    tracker = _numbered_tracker("1-9", first)
    for t in [t for t in tracker.tracks if t.number in (1, 2, 3, 4, 5)]:
        _pot(tracker, t, 12)
    rack = [((10.0 + 8 * k, 30.0), c, 0.05) for k, c in enumerate(colours)]
    for i in range(13, 26):
        tracker.update([_det(xy, family=fam, stripe=st) for xy, fam, st in rack], 1 / 30, i, i / 30.0)
    named = {t.number for t in tracker.tracks if t.time_since_update == 0}
    assert {1, 2, 3} <= named


# --------------------------------------------------------------------------
# Strikes
# --------------------------------------------------------------------------


class _Ball:
    """What the event detector and the shot segmenter read off a track."""

    def __init__(self, tid, xy, label="#1"):
        from billiards.kalman import BallKalman

        self.track_id = tid
        self.state = TrackState.CONFIRMED
        self.kf = BallKalman(tuple(xy))
        self._speed = 0.0
        self.age = 100
        self.trail = []
        self.label = label
        self.ball_type = "cue" if label == "CUE" else "solid"
        self.velocity = np.zeros(2)
        self.time_since_update = 0

    @property
    def speed(self):
        return self._speed

    @property
    def last_observed_xy(self):
        return tuple(self.kf.position)

    def at(self, frame, xy, speed, fps=30.0):
        if self.trail:
            step = np.asarray(xy, dtype=np.float64) - np.asarray(self.trail[-1].table_xy)
            self.velocity = step * fps / max(1, frame - self.trail[-1].frame)
        self.kf.x[:2] = xy
        self._speed = speed
        self.trail.append(TrackSample(frame, frame / fps, tuple(map(float, xy)), (0.0, 0.0), speed, True))


def _detector():
    return EventDetector(Config(), _table(), fps=30.0)


def _strikes(events):
    return [e for e in events if e.type is EventType.BALL_STRUCK]


def test_balls_far_apart_set_off_together_are_the_picture_not_play():
    """Every dissolve of the 2026 Premier League final 'struck' 3-8 balls at
    once, opened a shot, and one potted the 9."""
    det = _detector()
    a, b = _Ball(1, (20.0, 10.0)), _Ball(2, (80.0, 40.0))
    for f in range(3):
        a.at(f, (20.0, 10.0), 0.0)
        b.at(f, (80.0, 40.0), 0.0)
        det.step([a, b], f, f / 30.0)
    a.at(3, (20.5, 10.0), 60.0)
    b.at(3, (80.5, 40.0), 60.0)
    assert _strikes(det.step([a, b], 3, 3 / 30.0)) == []
    assert det.picture_moves == 1


def test_a_ball_set_off_by_one_already_rolling_is_struck():
    """In play the second ball moves because the first hit it: the cue ball
    is rolling when the object ball sets off."""
    det = _detector()
    cue, ball = _Ball(1, (20.0, 10.0), "CUE"), _Ball(2, (80.0, 40.0))
    for f in range(3):
        cue.at(f, (20.0, 10.0), 0.0)
        ball.at(f, (80.0, 40.0), 0.0)
        det.step([cue, ball], f, f / 30.0)
    cue.at(3, (22.0, 10.0), 80.0)
    ball.at(3, (80.0, 40.0), 0.0)
    assert len(_strikes(det.step([cue, ball], 3, 3 / 30.0))) == 1
    cue.at(4, (24.0, 10.0), 80.0)
    ball.at(4, (81.0, 40.0), 60.0)
    assert len(_strikes(det.step([cue, ball], 4, 4 / 30.0))) == 1


def test_a_ball_seen_a_ball_width_away_at_once_was_struck():
    """Struck and into another ball between two sightings, the tripod key's
    cue ball jumped 69 px, and its filtered speed read 22 in/s."""
    det = _detector()
    cue = _Ball(1, (50.0, 25.0), "CUE")
    for f in range(3):
        cue.at(f, (50.0, 25.0), 0.0)
        det.step([cue], f, f / 30.0)
    cue.at(4, (56.0, 25.0), 0.8 * det._struck_speed)   # 6 in in 2 frames: 90 in/s
    assert len(_strikes(det.step([cue], 4, 4 / 30.0))) == 1


# --------------------------------------------------------------------------
# Shots
# --------------------------------------------------------------------------


def _struck(tid, frame, xy=(50.0, 25.0)):
    from billiards.events import Event

    return Event(type=EventType.BALL_STRUCK, frame=frame, t_s=frame / 30.0, table_xy=xy,
                 image_xy=(0.0, 0.0), track_ids=(tid,))


def test_a_shot_in_which_nothing_went_anywhere_is_dropped():
    """Racked balls, split out of their cluster a pixel or two off each
    frame, read 5-40 in/s without going anywhere."""
    seg = ShotSegmenter(Config(), fps=30.0)
    ball = _Ball(1, (50.0, 25.0))
    rng = np.random.default_rng(0)
    for f in range(40):
        ball.at(f, np.array([50.0, 25.0]) + rng.normal(0, 0.3, 2), 20.0)
        seg.step([ball], [_struck(1, f)] if f == 1 else [], f, f / 30.0)
    assert seg.shots == [] and seg._current is None


def test_a_shot_that_moves_a_ball_is_kept_and_ends_by_the_clock():
    """At rest for 0.4 s of play, not 0.4 s of the file's frames: a
    broadcast that repeats every other frame is measured 30 times a second."""
    seg = ShotSegmenter(Config(), fps=60.0)
    cue = _Ball(1, (20.0, 25.0), "CUE")
    ended = None
    for k in range(60):                      # measured every other frame of 60 fps
        f = 2 * k
        x = 20.0 + min(k, 20) * 2.0
        cue.at(f, (x, 25.0), 60.0 if k < 20 else 0.0, fps=60.0)
        done = seg.step([cue], [_struck(1, f)] if k == 1 else [], f, f / 60.0)
        if done is not None:
            ended = done
    assert ended is not None and len(seg.shots) == 1
    assert ended.end_t_s - 40 / 60.0 < 0.5


# --------------------------------------------------------------------------
# Shots against an answer key
# --------------------------------------------------------------------------


def test_reported_shots_are_matched_to_the_marked_ones():
    marked = [{"frame": 100, "potted": [4]}, {"frame": 400, "potted": []}, {"frame": 900, "potted": [2]}]
    reported = [
        {"start_frame": 110, "potted": ["4"]},          # found, pot right
        {"start_frame": 250, "potted": []},             # nothing marked near: extra
        {"start_frame": 395, "potted": ["9"]},          # found, pot wrong
        {"start_frame": 905, "potted": ["2", "CUE"]},   # found; the scratch is left out
    ]
    got = real_eval.score_shots(marked, reported, fps=30.0)
    assert got["shots_found"] == 3 and got["shots_extra"] == 1 and got["pots_right"] == 2


# --------------------------------------------------------------------------
# A ball in a pocket's jaws, and a cue ball that jumps
# --------------------------------------------------------------------------


def test_a_ball_passed_over_is_not_hit():
    """``albin_fedor``'s first shot jumps the cue ball over the 6 to pot the 4;
    on the screen its path crosses the 6, which did not move."""
    det = _detector()
    cue, six = _Ball(1, (30.0, 25.0), "CUE"), _Ball(2, (40.0, 25.0))
    for f in range(3):
        cue.at(f, (30.0, 25.0), 0.0)
        six.at(f, (40.0, 25.0), 0.0)
        det.step([cue, six], f, f / 30.0)
    events = []
    for k, f in enumerate(range(3, 20)):
        cue.at(f, (32.0 + 4.0 * k, 25.2), 120.0)
        six.at(f, (40.0, 25.0), 0.0)
        events += det.step([cue, six], f, f / 30.0)
    assert [e for e in events if e.type is EventType.COLLISION] == []


def test_a_ball_hit_from_rest_is_hit_once_it_moves():
    det = _detector()
    cue, ball = _Ball(1, (30.0, 25.0), "CUE"), _Ball(2, (40.0, 25.0))
    for f in range(3):
        cue.at(f, (30.0, 25.0), 0.0)
        ball.at(f, (40.0, 25.0), 0.0)
        det.step([cue, ball], f, f / 30.0)
    events = []
    for k, f in enumerate(range(3, 12)):
        cue.at(f, (32.0 + 3.0 * min(k, 2), 25.0), 90.0 if k <= 2 else 0.0)
        ball.at(f, (40.0 + 3.0 * max(0, k - 2), 25.0), 0.0 if k <= 2 else 90.0)
        events += det.step([cue, ball], f, f / 30.0)
    hits = [e for e in events if e.type is EventType.COLLISION]
    assert len(hits) == 1 and hits[0].frame <= 6


def test_a_ball_that_vanishes_in_a_pockets_mouth_is_potted_there():
    """Followed into the jaws, a ball's last sightings jitter as it drops, and
    coasting on them the ceiling camera's 1 left the pocket and was lost."""
    tracker = MultiObjectTracker(Config(), _table(), fps=30.0)
    # Along the rail into the side pocket: coasting on, it would roll past.
    path = [(40.0 + 1.5 * i, 1.2) for i in range(8)]
    for i, xy in enumerate(path):
        tracker.update([_det(xy, family="red")], 1 / 30, i, i / 30.0)
    for i in range(len(path), len(path) + 6):
        tracker.update([], 1 / 30, i, i / 30.0)
    dead = list(tracker.limbo) + list(tracker.finished)
    assert [t.death_reason for t in dead] == ["potted"]


def test_an_event_just_before_a_late_shot_belongs_to_it():
    """The tripod key's shot that potted the 9 opened 1.5 s late; the pot,
    dated to when the 9 vanished, fell before it."""
    seg = ShotSegmenter(Config(), fps=30.0)
    ball = _Ball(1, (20.0, 25.0), "CUE")
    for f in range(0, 30):
        ball.at(f, (20.0 + 2.0 * max(0, f - 10), 25.0), 60.0 if f >= 10 else 0.0)
        seg.step([ball], [_struck(1, f)] if f == 10 else [], f, f / 30.0)
    from billiards.events import Event

    pot = Event(type=EventType.POT, frame=4, t_s=4 / 30.0, table_xy=(0.0, 0.0), image_xy=(0.0, 0.0),
                track_ids=(7,))
    seg.step([ball], [pot], 30, 1.0)
    assert seg._current.potted == ["#7"] and seg._current.start_frame == 4


@pytest.mark.slow
@pytest.mark.skipif(not ballnet.MODEL_PATH.exists(), reason="no trained model in billiards/models/")
def test_the_4_hanging_in_albin_fedors_corner_pocket_is_found():
    """It sat half an inch from the pocket's centre, inside the disc blanked
    out round every pocket, and was never seen."""
    from billiards import RunOptions, build_pipeline
    from billiards.video import read_frames

    clip = Path(__file__).resolve().parents[1] / "albin_fedor.mp4"
    cfg = Config().apply_preset()
    pipe, _, _ = build_pipeline(cfg, RunOptions(video=str(clip)))
    result = None
    for i, frame in read_frames(str(clip), 0, 6, cfg.max_frame_width):
        result = pipe.process(frame, i, annotate=False)
    assert any(abs(d.centre_image[0] - 366) < 12 and abs(d.centre_image[1] - 541) < 12
               for d in result.detections)
