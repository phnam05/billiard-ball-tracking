"""End-to-end tests against a synthetic clip with known ground truth.

These are the tests that actually answer "is the tracking robust?".  A short
break is rendered from a virtual camera at a realistic angle, complete with
perspective, a lighting gradient, sensor noise, a cue stick and a tightly racked
cluster, and the tracker is scored against the simulator's exact ball positions.

The thresholds below are deliberately a little looser than the numbers the
pipeline currently achieves, so they catch regressions without failing on
harmless noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from billiards import Config, RunOptions, run
from billiards.kalman import BallKalman


# --------------------------------------------------------------------------
# Adaptive filter behaviour
# --------------------------------------------------------------------------


def test_stationary_ball_does_not_drift_into_motion():
    """The failure that a fixed, large process noise guarantees.

    With process noise tuned loose enough to follow a collision, detection
    jitter alone makes a motionless ball read as travelling at 15-20 in/s, and
    every downstream speed rule -- shot detection, the stationary cut-off, the
    trail -- becomes meaningless.
    """
    rng = np.random.default_rng(11)
    truth = np.array([40.0, 25.0])
    kf = BallKalman(tuple(truth), meas_std_in=0.2)
    for _ in range(120):
        kf.predict(1 / 30)
        kf.update(tuple(truth + rng.normal(0, 0.2, 2)))
    assert kf.speed < 4.0
    assert np.linalg.norm(kf.position - truth) < 0.3


def test_filter_follows_a_cushion_bounce_quickly():
    """A bounce reverses the velocity between two frames; the filter has to
    follow within a frame or two rather than ploughing through the rail."""
    kf = BallKalman((50.0, 25.0), meas_std_in=0.2)
    dt = 1 / 120
    # Approach the cushion at 120 in/s.
    x = 50.0
    for _ in range(40):
        x += 120.0 * dt
        kf.predict(dt)
        kf.update((x, 25.0))
    assert kf.velocity[0] > 80.0

    # Reverse.
    for _ in range(8):
        x -= 120.0 * dt
        kf.predict(dt)
        kf.update((x, 25.0))
    assert kf.velocity[0] < -40.0, "filter did not turn around at the cushion"


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory) -> dict:
    from make_synthetic_clip import generate

    out = tmp_path_factory.mktemp("clip")
    video = out / "break.mp4"
    gt = out / "gt.csv"
    info = generate(
        video, seed=0, fps=30.0, duration_s=4.0, width=960, height=540,
        ground_truth_path=gt,
    )
    return {"video": str(video), "gt": str(gt), "dir": out, "info": info}


@pytest.fixture(scope="module")
def retimed_clip(tmp_path_factory) -> dict:
    """The same break, captured the way the sample broadcasts were: 25 fps
    content in a 37.5 fps file, repeated, late, and one frame in eight missed."""
    from make_synthetic_clip import generate

    out = tmp_path_factory.mktemp("retimed")
    video = out / "break.mp4"
    gt = out / "gt.csv"
    info = generate(
        video, seed=0, fps=25.0, duration_s=4.0, width=960, height=540,
        ground_truth_path=gt, container_fps=37.5, drop_rate=0.12,
        capture_jitter=0.6,
    )
    return {"video": str(video), "gt": str(gt), "dir": out, "info": info}


@pytest.mark.slow
def test_calibration_finds_the_table(synthetic_clip):
    from billiards.pipeline import build_pipeline

    cfg = Config()
    pipeline, calib, info = build_pipeline(cfg, RunOptions(video=synthetic_clip["video"]))

    assert calib.frames_used >= 0.8 * calib.frames_attempted
    assert calib.corner_spread_px < 12.0
    # The sample broadcasts' blue-grey cloth: OpenCV hue 109 on all three.
    assert 95 < calib.cloth.hue < 120
    # Sanity: a 9ft table filling most of a 960px frame is ~7-11 px per inch.
    assert 4.0 < calib.table.mean_px_per_inch() < 20.0


@pytest.mark.slow
def test_calibration_finds_a_green_table(tmp_path):
    """The synthetic cloth was green until 23 Sep 2026; green still has to work."""
    from billiards.pipeline import build_pipeline
    from make_synthetic_clip import generate

    video = tmp_path / "green.mp4"
    generate(video, seed=0, fps=30.0, duration_s=1.0, width=960, height=540, cloth="green")
    _, calib, _ = build_pipeline(Config(), RunOptions(video=str(video)))
    assert calib.frames_used >= 0.8 * calib.frames_attempted
    assert calib.corner_spread_px < 12.0
    assert 30 < calib.cloth.hue < 95


@pytest.mark.slow
def test_tracking_accuracy_against_ground_truth(synthetic_clip, tmp_path):
    from evaluate import evaluate, load_ground_truth, load_tracks, load_visibility

    csv_path = tmp_path / "tracks.csv"
    cfg = Config()
    summary = run(
        cfg,
        RunOptions(
            video=synthetic_clip["video"],
            export_csv=str(csv_path),
            progress_every=0,
        ),
    )
    assert summary["frames_processed"] > 100

    report = evaluate(
        load_ground_truth(Path(synthetic_clip["gt"])),
        load_tracks(csv_path),
        gate_in=2.25,
        gt_visibility=load_visibility(Path(synthetic_clip["gt"])),
    )

    # Scored on balls at least half in view: from behind an end rail a static
    # rack hides most of itself, and no tracker can report what is not in the
    # picture.  (Until 23 Sep 2026 the synthetic camera looked across the
    # table from a view no real camera can produce, balls were drawn as discs
    # painted on the cloth, and these bars were set against that.)
    #
    # Measured 26 Sep 2026: recall 0.797, precision 0.992, MOTA 0.783, 7 ID
    # switches.  Since 23 Sep evening the simulator draws the sample
    # broadcasts' own ball colours on their blue-grey cloth, and most of the
    # misses are one ball: the blue stripe, whose band the cloth mask takes
    # for cloth (recall 0.31) -- a cloth-coloured ball, hard by construction.
    assert report["recall"] > 0.77, report
    assert report["precision"] > 0.95, report
    assert report["id_switches"] <= 9, report
    assert report["mota"] > 0.75, report
    # Half a ball radius is the accuracy that makes contact points meaningful.
    assert report["position_error_in"]["median"] < 0.55, report
    # A constant-rate file never repeats a frame while a ball rolls, so the
    # scene clock must stay out of the way.
    assert summary["clock"]["retimed"] is False


@pytest.mark.slow
def test_speeds_are_right_on_a_screen_recorded_clip(retimed_clip, tmp_path):
    """Believing the file's clock on such a clip puts speeds ~10% off, and one
    in twenty off by 48 in/s or more, while positions stay right.  Measured on
    this clip: 10.2% median / 48 in/s at the 95th percentile with the file's
    clock, 5.1% / 26 in/s with the scene's."""
    from evaluate import evaluate, load_ground_truth, load_speeds, load_tracks, load_visibility

    csv_path = tmp_path / "tracks.csv"
    summary = run(
        Config(),
        RunOptions(video=retimed_clip["video"], export_csv=str(csv_path), progress_every=0),
    )
    clock = summary["clock"]
    assert clock["retimed"] is True, clock
    # Read 57.9 fps before the rate was bounded and the filters' velocities
    # followed its changes (see billiards/clock.py).
    assert clock["source_fps"] == pytest.approx(25.0), clock

    gt = Path(retimed_clip["gt"])
    report = evaluate(
        load_ground_truth(gt), load_tracks(csv_path), gate_in=2.25,
        gt_speeds=load_speeds(gt, "ball"), track_speeds=load_speeds(csv_path, "track_id"),
        gt_visibility=load_visibility(gt),
    )
    # Measured 26 Sep 2026: 11.1%.  The rate estimate sits at 26-27 fps for
    # most of these four seconds before it settles on 25, and the speeds
    # follow it; on the report's longer clip the same capture scores 4.9%.
    assert report["speed_error"]["median_relative"] < 0.13, report["speed_error"]
    # 25 fps content with one frame in eight missing, at 960 px: measured 0.768.
    assert report["mota"] > 0.74, report


@pytest.mark.slow
def test_at_most_one_cue_ball_and_one_eight(synthetic_clip, tmp_path):
    """A table has exactly one of each, so the labels must too.

    Classifying each ball on its own appearance gave two "cue balls" and four
    "8 balls" on a real clip -- grey cloth pushes several balls into "dark and
    colourless" at once.  The roles are assigned across the whole set instead.
    """
    import csv as _csv
    from collections import Counter, defaultdict

    csv_path = tmp_path / "tracks.csv"
    run(
        Config(),
        RunOptions(
            video=synthetic_clip["video"],
            export_csv=str(csv_path),
            progress_every=0,
        ),
    )

    per_frame = defaultdict(Counter)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["state"] == "confirmed":
                per_frame[int(row["frame"])][row["ball_type"]] += 1

    assert per_frame, "no confirmed tracks at all"
    for frame, counts in per_frame.items():
        assert counts["cue"] <= 1, f"frame {frame}: {counts}"
        assert counts["eight"] <= 1, f"frame {frame}: {counts}"
    # And the cue ball really was found on this clip.
    assert any(c["cue"] == 1 for c in per_frame.values())


@pytest.mark.slow
def test_no_phantom_balls_at_the_pockets(synthetic_clip, tmp_path):
    """Pockets are dark, round and ball-sized, and they never move.

    Without explicit exclusion every pocket becomes a permanent phantom track,
    which is both wrong and noisy -- it was four extra "balls" on every frame.
    """
    import csv
    from collections import defaultdict

    csv_path = tmp_path / "tracks.csv"
    run(
        Config(),
        RunOptions(
            video=synthetic_clip["video"],
            export_csv=str(csv_path),
            progress_every=0,
        ),
    )

    per_frame = defaultdict(int)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["state"] == "confirmed":
                per_frame[int(row["frame"])] += 1

    counts = sorted(per_frame.values())
    # Nine balls are on the table; allow a little slack for a split cluster,
    # but nothing like the 13 that phantom pockets produced.
    assert max(counts) <= 11, f"too many simultaneous tracks: {counts[-5:]}"


@pytest.mark.slow
def test_a_break_is_reported_as_one_shot(synthetic_clip, tmp_path):
    """A flat event list is accurate but unreadable; a shot is what a player
    asks for.  The synthetic clip is exactly one break, so it must come back as
    exactly one shot, opened by the cue ball."""
    summary = run(
        Config(),
        RunOptions(
            video=synthetic_clip["video"],
            export_json=str(tmp_path / "run.json"),
            progress_every=0,
        ),
    )
    shots = summary["shot_log"]
    assert len(shots) == 1, [s["summary"] for s in shots]
    shot = shots[0]
    assert shot["opener"] == "CUE", shot
    assert shot["collisions"] >= 1, shot
    assert shot["duration_s"] and shot["duration_s"] > 0.5, shot
    assert shot["peak_speed_in_s"] > 100.0, shot


def _struck_events(detector, track, speeds, start_frame=0):
    """Feed a speed profile through the detector and collect the strikes."""
    struck = []
    for i, s in enumerate(speeds):
        track._speed = s
        # Well past the new-track age gate; these tests are about the
        # thresholds, not about a filter that has not settled yet.
        track.age = 100 + i
        for e in detector.step([track], start_frame + i, (start_frame + i) / 30.0):
            if e.type.value == "ball_struck":
                struck.append(e)
    return struck


class _FakeTrack:
    def __init__(self) -> None:
        from billiards.kalman import BallKalman
        from billiards.track import TrackState

        self.track_id = 1
        self.state = TrackState.CONFIRMED
        self.kf = BallKalman((50.0, 25.0))
        self._speed = 0.0
        self.age = 100
        self.velocity = np.zeros(2)
        self.time_since_update = 0

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def last_observed_xy(self):
        return tuple(self.kf.position)


def _detector():
    from billiards.events import EventDetector
    from billiards.geometry import TableModel

    table = TableModel(
        np.array([[300.0, 180.0], [980.0, 180.0], [1180.0, 600.0], [100.0, 600.0]]),
        length_in=100.0, width_in=50.0, ball_diameter_in=2.25,
    )
    return EventDetector(Config(), table, fps=30.0)


def test_a_ball_accelerating_through_the_band_is_still_reported_as_struck():
    """Regression: with one *low* threshold for the remembered state, a ball
    ramping up set "already moving" on the way and the strike was never
    reported.  A clip containing five pots produced zero shots."""
    detector = _detector()
    lo, hi = detector._rest_speed, detector._struck_speed
    ramp = [0.0, 0.5 * (lo + hi), hi * 3.0]
    assert len(_struck_events(detector, _FakeTrack(), ramp)) == 1


def test_a_wobbling_speed_does_not_fire_repeatedly():
    """Regression: with one threshold of any value, a ball whose estimated
    speed wobbles across it fires on every upward crossing -- one ball emitted
    eight strikes in half a second."""
    detector = _detector()
    hi = detector._struck_speed
    # Struck once, then jittering either side of the threshold while it rolls.
    profile = [0.0, hi * 2.5] + [hi * 0.9, hi * 1.4] * 6
    assert len(_struck_events(detector, _FakeTrack(), profile)) == 1


def test_a_ball_that_truly_stops_can_be_struck_again():
    detector = _detector()
    lo, hi = detector._rest_speed, detector._struck_speed
    profile = [0.0, hi * 2.5, hi * 1.2, lo * 0.5, 0.0, hi * 2.5]
    assert len(_struck_events(detector, _FakeTrack(), profile)) == 2


def test_contact_is_found_even_when_the_ball_steps_over_it():
    """Two balls are in contact across a shell a quarter of an inch thick, and
    a cue ball crossing the table covers three or four inches between frames,
    so sampling the gap at the end of the frame finds contact about one time in
    fifteen.  On fedor_shot.mp4 the gap read 3.72 in on one frame and 2.53 in
    on the next, against a 2.52 in threshold: the shot potted a ball and
    reported no collision."""
    from billiards.events import closest_approach

    d = 2.25
    # The cue ball runs along y=0 at eight inches a frame and clips a ball
    # sitting 2.3 in off its line.  Sampled at either end of the frame it is
    # 4.6 in away; halfway through it is touching.
    gap = np.array([4.0, 2.3])
    change = np.array([-8.0, 0.0])
    s, dist = closest_approach(gap, change)
    assert s == pytest.approx(0.5)
    assert dist == pytest.approx(2.3), "contact during the frame must be found"
    assert min(
        np.linalg.norm(gap), np.linalg.norm(gap + change)
    ) > 1.25 * d, "...and neither endpoint would have found it"


def test_a_grazing_pass_is_not_a_contact():
    """The other half of the gate: measuring over the whole frame must not turn
    every near miss into a collision.  Ground truth puts real near misses
    1.46-1.50 diameters apart."""
    from billiards.events import closest_approach

    gap = np.array([6.0, 4.0])
    change = np.array([-12.0, 0.0])  # sails past, four inches to the side
    _, dist = closest_approach(gap, change)
    assert dist == pytest.approx(4.0)
    assert dist > 1.25 * 2.25


def test_two_balls_frozen_together_never_collide():
    """They are as close as it gets, forever, so proximity alone would fire on
    every frame for the rest of the clip.  The gap has to be shrinking."""
    detector = _detector()
    a, b = _FakeTrack(), _FakeTrack()
    b.track_id = 2
    b.kf = type(a.kf)((50.0 + 2.25, 25.0))

    fired = []
    for i in range(30):
        fired += [
            e for e in detector.step([a, b], i, i / 30.0, 1 / 30)
            if e.type.value == "collision"
        ]
    assert fired == []


def _drive(detector, paths, fps=30.0):
    """Feed scripted raw paths {track id: [(x, y), ...]} through the detector."""
    from billiards.kalman import BallKalman

    tracks = {}
    for tid in paths:
        t = _FakeTrack()
        t.track_id = tid
        tracks[tid] = t
    events = []
    n = max(len(p) for p in paths.values())
    for i in range(n):
        for tid, path in paths.items():
            p = path[min(i, len(path) - 1)]
            q = path[max(0, min(i, len(path) - 1) - 1)]
            tracks[tid].kf = BallKalman(p)
            tracks[tid].velocity = (np.array(p) - np.array(q)) * fps
            tracks[tid]._speed = float(np.linalg.norm(tracks[tid].velocity))
        events += detector.step(list(tracks.values()), i, i / fps, 1.0 / fps)
    return events


def _bounce_path(apex_y=1.125, speed=60.0, fps=30.0, frames=14):
    """Rolling at 45 degrees into the y = 0 rail and back out."""
    step = speed / fps / np.sqrt(2.0)
    # Well clear of the side pocket at x = 50, where a turn-round is not a cushion.
    return [(15.0 + step * k, apex_y + abs(12.0 - step * k)) for k in range(frames)]


def test_a_bounce_off_a_rail_is_one_cushion():
    """Found from the raw path, not the filter: the filter turns the corner over
    two or three frames, by which time the ball is inches off the rail, and a
    cushion detector reading its velocity found 2 of 23 contacts on the
    synthetic break."""
    events = _drive(_detector(), {1: _bounce_path()})
    cushions = [e for e in events if e.type.value == "cushion"]
    assert len(cushions) == 1, events
    assert cushions[0].detail["rail_distance_in"] < 2.5
    assert cushions[0].table_xy[1] < 2.5


def test_turning_round_next_to_another_ball_is_not_a_cushion():
    """A ball that turns round beside another ball was turned by that ball."""
    path = _bounce_path(apex_y=3.5)
    apex = min(path, key=lambda p: p[1])
    beside = [(apex[0] + 0.6, apex[1] - 2.2)] * len(path)
    events = _drive(_detector(), {1: path, 2: beside})
    assert not [e for e in events if e.type.value == "cushion"], events


def test_rolling_along_a_rail_is_not_a_cushion():
    path = [(20.0 + 1.5 * k, 1.6 + 0.08 * (-1) ** k) for k in range(20)]
    events = _drive(_detector(), {1: path})
    assert not [e for e in events if e.type.value == "cushion"], events


def test_a_ball_that_stopped_short_of_a_rail_does_not_bounce_off_it_later():
    """Rolling toward the far rail, it stops 39 in short; two seconds later it
    is knocked back the way it came.  Steps too small to class leave the
    "approaching" state alone, so this read as a bounce off a rail 39 in away
    (fedor_shot, frame 303).  Seen standing that long, it approaches nothing."""
    fps = 30.0
    rolling = [(40.0 + 1.5 * k, 20.0) for k in range(8)]           # toward x = L
    resting = [rolling[-1]] * 60                                     # 2 s, seen still
    knocked = [(rolling[-1][0] - 2.0 * k, 20.0) for k in range(1, 8)]
    events = _drive(_detector(), {1: rolling + resting + knocked}, fps=fps)
    assert not [e for e in events if e.type.value == "cushion"], events


def test_a_stun_shot_is_one_collision():
    """The cue ball stops dead and the object ball takes its speed: both paths
    turn a corner at the moment of contact, a ball's width apart."""
    fps, v = 30.0, 60.0
    contact = 6
    cue = [(30.0 + v / fps * min(k, contact), 25.0) for k in range(16)]
    touch = cue[contact][0] + 2.25
    obj = [(touch + v / fps * max(0, k - contact), 25.0) for k in range(16)]
    events = _drive(_detector(), {1: cue, 2: obj}, fps=fps)
    collisions = [e for e in events if e.type.value == "collision"]
    assert len(collisions) == 1, events
    assert set(collisions[0].track_ids) == {1, 2}
    assert abs(collisions[0].frame - contact) <= 1


@pytest.mark.slow
def test_events_are_detected_on_a_break(synthetic_clip, tmp_path):
    summary = run(
        Config(),
        RunOptions(
            video=synthetic_clip["video"],
            export_json=str(tmp_path / "run.json"),
            progress_every=0,
        ),
    )
    events = summary["events"]
    assert events.get("collision", 0) >= 1, summary["event_log"]
    assert summary["tracks_created"] >= 8


@pytest.mark.slow
def test_output_video_is_written_and_playable(synthetic_clip, tmp_path):
    """The original wrote a video that could not be decoded: the writer was
    sized for the source frames but fed resized ones, and it wrote the frame
    *before* anything was drawn on it."""
    import cv2

    out = tmp_path / "annotated.mp4"
    summary = run(
        Config(),
        RunOptions(video=synthetic_clip["video"], output=str(out), progress_every=0),
    )
    assert out.exists() and out.stat().st_size > 10_000

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0.9 * summary["frames_processed"]
    finally:
        cap.release()


@pytest.mark.slow
def test_tracking_pauses_when_the_table_leaves_the_view(synthetic_clip, tmp_path):
    """Broadcast footage cuts to replays, crowd shots and player close-ups.

    With a stale homography still applied, a cut produced dozens of "balls"
    sitting on spectators and trajectories drawn between them.  Nothing at all
    is the correct output for a frame with no table in it.
    """
    import cv2

    src = cv2.VideoCapture(synthetic_clip["video"])
    frames = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        frames.append(f)
    src.release()
    assert frames

    h, w = frames[0].shape[:2]
    rng = np.random.default_rng(5)
    # A "cut": a busy, cloth-free scene of the same size.
    cut = [
        rng.integers(0, 255, (h, w, 3), dtype=np.uint8) // 2 + 40
        for _ in range(45)
    ]

    mixed = tmp_path / "mixed.mp4"
    writer = cv2.VideoWriter(
        str(mixed), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )
    for f in frames + cut:
        writer.write(f)
    writer.release()

    csv_path = tmp_path / "tracks.csv"
    summary = run(
        Config(),
        RunOptions(video=str(mixed), export_csv=str(csv_path), progress_every=0),
    )
    assert summary["frames_view_lost"] > 10, summary

    import csv as _csv
    from collections import Counter

    per_frame = Counter()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            per_frame[int(row["frame"])] += 1

    cut_start = len(frames)
    # A few frames of patience at the boundary is expected; deep into the cut
    # there must be nothing at all.
    for f in range(cut_start + 15, cut_start + len(cut)):
        assert per_frame.get(f, 0) == 0, f"tracks reported on frame {f} with no table"


@pytest.mark.slow
def test_a_dissolve_does_not_invent_balls(synthetic_clip, tmp_path):
    """The case cloth coverage is blind to.

    A broadcast crossfades between two shots of the *same* sport, so the
    incoming angle is mostly cloth as well and the outgoing bed polygon keeps
    passing the coverage test most of the way through the transition.  On a
    real clip that bought twelve frames in which the crowd, the rails and a
    second table were all inside a stale bed polygon, and those twelve frames
    created thirty phantom tracks -- half of everything the clip produced.

    Early in a fade the table really is still there and tracking it is right,
    so what this pins is not "stop immediately" but "create nothing": the
    dissolve must not leave the run with more balls than the table has.  It
    does not care which guard gets there first -- on this clip, deliberately
    lower in contrast than a real cut, the detector's own tests are what hold
    (35 tracks without them, 9 with); ``test_a_cut_is_seen_the_frame_it_starts``
    covers the signal meant for a full-contrast transition.
    """
    import cv2

    src = cv2.VideoCapture(synthetic_clip["video"])
    frames = []
    while True:
        ok, f = src.read()
        if not ok:
            break
        frames.append(f)
    src.release()
    assert frames

    clean = run(
        Config(),
        RunOptions(video=synthetic_clip["video"], progress_every=0),
    )

    h, w = frames[0].shape[:2]
    rng = np.random.default_rng(3)
    # A busy, cloth-free scene, deliberately lower in contrast than a real cut
    # to a crowd -- the transition this has to survive is the hard one.
    after = rng.integers(0, 255, (h, w, 3), dtype=np.uint8) // 2 + 40

    last = frames[-1]
    fade = [
        cv2.addWeighted(last, 1.0 - a, after, a, 0.0)
        for a in np.linspace(0.0, 1.0, 12)
    ]
    hold = [after.copy() for _ in range(25)]

    mixed = tmp_path / "dissolve.mp4"
    writer = cv2.VideoWriter(str(mixed), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in frames + fade + hold:
        writer.write(f)
    writer.release()

    csv_path = tmp_path / "tracks.csv"
    summary = run(
        Config(),
        RunOptions(video=str(mixed), export_csv=str(csv_path), progress_every=0),
    )

    assert summary["tracks_created"] <= clean["tracks_created"] + 2, (
        "the dissolve created {} tracks where the clip alone creates {}".format(
            summary["tracks_created"], clean["tracks_created"]
        )
    )

    import csv as _csv
    from collections import Counter

    per_frame = Counter()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            per_frame[int(row["frame"])] += 1

    # ...and once the table is gone outright, nothing at all.
    hold_start = len(frames) + len(fade)
    for f in range(hold_start + 5, hold_start + len(hold)):
        assert per_frame.get(f, 0) == 0, f"tracks reported on frame {f} with no table"


class TestFailsClearly:
    """A tool that needs no tuning still has to say what went wrong when it
    genuinely cannot proceed.  These pin the wording, because a vague error is
    exactly what sends someone back to editing thresholds by hand."""

    def test_missing_file(self):
        from billiards.cli import main

        assert main(["track", "definitely_not_here.mp4"]) == 1

    def test_no_table_in_the_video(self, tmp_path):
        import cv2

        rng = np.random.default_rng(1)
        path = tmp_path / "noise.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360)
        )
        for _ in range(40):
            writer.write(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
        writer.release()

        with pytest.raises(RuntimeError, match="Table calibration failed"):
            run(Config(), RunOptions(video=str(path), progress_every=0))

    def test_start_time_past_the_end(self, synthetic_clip):
        with pytest.raises(RuntimeError, match="Could not read any frames"):
            run(
                Config(),
                RunOptions(
                    video=synthetic_clip["video"],
                    start_frame=10_000,
                    progress_every=0,
                ),
            )

    def test_a_v1_style_config_key_is_named_in_the_error(self):
        # Someone porting an old config will reach for the HSV bounds first.
        with pytest.raises(ValueError, match="lower_bound"):
            Config.from_dict({"detector": {"lower_bound": [30, 20, 200]}})


@pytest.mark.slow
def test_manual_table_corners_are_respected(synthetic_clip):
    from billiards.pipeline import build_pipeline

    corners = [[100, 100], [800, 100], [900, 500], [50, 500]]
    _, calib, _ = build_pipeline(
        Config(), RunOptions(video=synthetic_clip["video"], table_corners=corners)
    )
    assert np.allclose(
        sorted(calib.table.corners_image.tolist()), sorted([list(map(float, c)) for c in corners])
    )
