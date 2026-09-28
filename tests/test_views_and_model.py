"""What changed on 28 Sep 2026 evening: a ball's colour per camera, someone
in front of the lens, a cloth with no colour, and the ball model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards import balls as ballnum
from billiards import ballnet
from billiards import Config, RunOptions
from billiards.detect import ColorSignature, Detection, colour_distance_matrix
from billiards.geometry import TableModel
from billiards.table import ClothModel
from billiards.track import MultiObjectTracker


def _table() -> TableModel:
    corners = np.array([[300.0, 180.0], [980.0, 180.0], [1180.0, 600.0], [100.0, 600.0]])
    return TableModel(corners, length_in=100.0, width_in=50.0, ball_diameter_in=2.25)


def _det(xy, lab) -> Detection:
    return Detection(
        centre_image=(0.0, 0.0), centre_table=xy, radius_px=10.0, area_ratio=1.0,
        circularity=0.9, signature=ColorSignature(lab=np.array(lab, dtype=np.float64)),
        rim_contrast=40.0,
    )


# One ball as two cameras show it: further apart than the colour gate, as the
# same balls were on the 2026 US Open, but not twice as far.
THIS_CAMERA = [150.0, 150.0, 170.0]
OTHER_CAMERA = [150.0, 174.0, 217.0]


def _followed(tracker, frames=12):
    for i in range(frames):
        tracker.update([_det((50.0, 25.0), THIS_CAMERA)], 1 / 30, i, i / 30.0)
    return tracker.tracks[0]


def test_the_two_colours_are_further_apart_than_the_gate():
    cfg = Config()
    d = colour_distance_matrix([ColorSignature(lab=np.array(THIS_CAMERA))],
                               [ColorSignature(lab=np.array(OTHER_CAMERA))])[0, 0]
    assert cfg.tracker.max_color_distance < d < 1.8 * cfg.tracker.max_color_distance


def test_a_ball_set_aside_is_found_again_through_another_cameras_colours():
    cfg = Config()
    tracker = MultiObjectTracker(cfg, _table(), fps=30.0)
    ball = _followed(tracker)
    tracker.set_aside(12)
    tracker.set_view(1)
    for i in range(13, 20):
        tracker.update([_det((50.4, 25.2), OTHER_CAMERA)], 1 / 30, i, i / 30.0)
    assert [t.track_id for t in tracker.tracks] == [ball.track_id]


def test_from_the_same_camera_that_colour_is_another_ball():
    cfg = Config()
    tracker = MultiObjectTracker(cfg, _table(), fps=30.0)
    ball = _followed(tracker)
    tracker.set_aside(12)
    for i in range(13, 20):
        tracker.update([_det((50.4, 25.2), OTHER_CAMERA)], 1 / 30, i, i / 30.0)
    assert ball.track_id not in [t.track_id for t in tracker.tracks]


def test_each_camera_keeps_its_own_colour_for_a_ball():
    cfg = Config()
    tracker = MultiObjectTracker(cfg, _table(), fps=30.0)
    ball = _followed(tracker)
    tracker.set_aside(12)
    tracker.set_view(1)
    for i in range(13, 40):
        tracker.update([_det((50.4, 25.2), OTHER_CAMERA)], 1 / 30, i, i / 30.0)

    def gap(lab):
        return colour_distance_matrix([ball.signature], [ColorSignature(lab=np.array(lab))])[0, 0]

    assert gap(OTHER_CAMERA) < gap(THIS_CAMERA)
    tracker.set_view(0)
    assert gap(THIS_CAMERA) < 1.0
    tracker.set_view(1)
    assert gap(OTHER_CAMERA) < gap(THIS_CAMERA)


def test_a_cloth_with_no_colour_is_not_told_apart_by_hue():
    """A ceiling camera's grey cloth, measured as hue 0 saturation 0: its edges,
    tinted blue by the lens, were not cloth, and the table came out 3 in short."""
    grey = ClothModel(hue=0.0, sat=0.0, val=138.0, hue_halfwidth=6.0, sat_halfwidth=40.0, val_halfwidth=70.0)
    tinted = np.full((4, 4, 3), (113, 10, 120), np.uint8)
    assert grey.mask(tinted).min() == 255
    green = ClothModel(hue=60.0, sat=150.0, val=138.0, hue_halfwidth=6.0, sat_halfwidth=40.0, val_halfwidth=70.0)
    blue = np.full((4, 4, 3), (100, 150, 138), np.uint8)
    assert green.mask(blue).max() == 0


@pytest.mark.slow
def test_someone_walking_past_the_lens_is_not_a_cut(tmp_path):
    """A player passing close to the camera repaints a third of the bed in a
    frame, and tracking stopped until the whole bed was clear: on the 2026 US
    Open, every ball on two keyframes of the answer key."""
    from make_synthetic_clip import generate

    from billiards.pipeline import run

    clip = tmp_path / "break.mp4"
    generate(clip, seed=3, fps=30.0, duration_s=4.0, width=960, height=540)
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    h, w = frames[0].shape[:2]
    for k in range(60, 90):
        # A dark figure in the foreground, over the near-left of the table.
        x = int(0.05 * w + 2 * (k - 60))
        cv2.ellipse(frames[k], (x + int(0.18 * w), h), (int(0.2 * w), int(0.55 * h)), 0, 0, 360, (28, 26, 30), -1)
    walked = tmp_path / "walked.mp4"
    out = cv2.VideoWriter(str(walked), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    csv_path = tmp_path / "tracks.csv"
    summary = run(Config(), RunOptions(video=str(walked), export_csv=str(csv_path), progress_every=0))
    assert summary["frames_occluded"] > 0, summary
    assert summary["frames_view_lost"] <= 3, summary
    import csv

    drawn = {int(r["frame"]) for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
    assert all(k in drawn for k in range(62, 90)), sorted(set(range(60, 90)) - drawn)


# --------------------------------------------------------------------------
# The ball model
# --------------------------------------------------------------------------


def test_without_the_model_file_the_rules_decide(monkeypatch, tmp_path):
    assert ballnet.load(tmp_path / "missing.onnx") is None
    monkeypatch.setenv("BILLIARDS_BALLNET", "off")
    assert ballnet.load() is None


@pytest.mark.skipif(not ballnet.MODEL_PATH.exists(), reason="no trained model in billiards/models/")
def test_the_model_scores_any_number_of_proposals():
    """OpenCV 5.0 crashed when the batch changed size between calls; the
    proposals go through in fixed batches."""
    net = ballnet.BallNet()
    frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    for n in (1, 17, 3, 40, 16):
        kind, family, stripe = net.score(frame, [(100.0 + 5 * i, 200.0) for i in range(n)], [9.0] * n)
        assert kind.shape == (n, 3) and family.shape == (n, len(ballnet.FAMILIES)) and stripe.shape == (n,)
        assert np.allclose(kind.sum(axis=1), 1.0, atol=1e-4)


def test_numbers_follow_the_models_colours_and_the_ball_set():
    fam = ballnum.MODEL_FAMILIES

    def evidence(name, p=0.9):
        v = np.full(len(fam), (1.0 - p) / (len(fam) - 1))
        v[fam.index(name)] = p
        return np.log(v)

    tracks = [evidence("pink"), evidence("purple"), evidence("yellow")]
    solid = [0.05, 0.05, 0.9]  # the yellow one is a stripe
    tv = ballnum.ball_specs("tv", range(1, 10))
    numbers = ballnum.assign_costs(ballnum.model_cost_matrix(tracks, solid, tv), tv, max_cost=2.3)
    assert numbers == [4, 5, 9]
    standard = ballnum.ball_specs("standard", range(1, 10))
    numbers = ballnum.assign_costs(ballnum.model_cost_matrix(tracks, solid, standard), standard, max_cost=2.3)
    assert numbers[0] is None and numbers[1] == 4 and numbers[2] == 9
