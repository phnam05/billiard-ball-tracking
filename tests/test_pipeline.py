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


@pytest.mark.slow
def test_calibration_finds_the_table(synthetic_clip):
    from billiards.pipeline import build_pipeline

    cfg = Config()
    pipeline, calib, info = build_pipeline(cfg, RunOptions(video=synthetic_clip["video"]))

    assert calib.frames_used >= 0.8 * calib.frames_attempted
    assert calib.corner_spread_px < 12.0
    # The cloth is green: OpenCV hue ~35-90.
    assert 30 < calib.cloth.hue < 95
    # Sanity: a 9ft table filling most of a 960px frame is ~7-11 px per inch.
    assert 4.0 < calib.table.mean_px_per_inch() < 20.0


@pytest.mark.slow
def test_tracking_accuracy_against_ground_truth(synthetic_clip, tmp_path):
    from evaluate import evaluate, load_ground_truth, load_tracks

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
    )

    assert report["recall"] > 0.85, report
    assert report["precision"] > 0.95, report
    assert report["id_switches"] <= 3, report
    assert report["mota"] > 0.82, report
    # Half a ball radius is the accuracy that makes contact points meaningful.
    assert report["position_error_in"]["median"] < 0.55, report


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


def test_ball_struck_uses_one_threshold_for_state_and_event():
    """Regression: the detector remembered "was moving" at the *stationary*
    speed but emitted at six times that, so a ball accelerating through the gap
    set the flag on the way up and its strike was never reported.  A clip with
    five pots produced zero shots."""
    from billiards.events import EventDetector
    from billiards.geometry import TableModel

    cfg = Config()
    table = TableModel(
        np.array([[300.0, 180.0], [980.0, 180.0], [1180.0, 600.0], [100.0, 600.0]]),
        length_in=100.0, width_in=50.0, ball_diameter_in=2.25,
    )
    detector = EventDetector(cfg, table)
    speed = detector._shot_speed
    assert speed > cfg.tracker.stationary_speed_in_s

    class _FakeTrack:
        def __init__(self) -> None:
            from billiards.kalman import BallKalman
            from billiards.track import TrackState

            self.track_id = 1
            self.state = TrackState.CONFIRMED
            self.kf = BallKalman((50.0, 25.0))
            self._speed = 0.0
            self.velocity = np.zeros(2)

        @property
        def speed(self) -> float:
            return self._speed

    track = _FakeTrack()
    # Frame 0: at rest. Frame 1: mid-ramp, between the two old thresholds.
    # Frame 2: clearly struck.  The event must fire.
    detector.step([track], 0, 0.0)
    track._speed = 0.5 * (cfg.tracker.stationary_speed_in_s + speed)
    detector.step([track], 1, 1 / 30)
    track._speed = speed * 3.0
    events = detector.step([track], 2, 2 / 30)
    assert any(e.type.value == "ball_struck" for e in events), events


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
