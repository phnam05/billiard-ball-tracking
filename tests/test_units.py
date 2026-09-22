"""Unit tests for the pieces that are easy to get subtly wrong."""

from __future__ import annotations

import json

import numpy as np
import pytest

from billiards.assignment import FORBIDDEN, _hungarian, associate, solve
from billiards.config import TABLE_PRESETS, Config
from billiards.detect import ColorSignature
from billiards.geometry import TableModel, order_corners, quad_from_contour
from billiards.kalman import BallKalman


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_presets_fill_dimensions():
    cfg = Config()
    cfg.table.preset = "snooker-12ft"
    cfg.apply_preset()
    assert cfg.table.length_in == pytest.approx(140.5)
    assert cfg.table.ball_diameter_in == pytest.approx(2.07)


def test_explicit_dimension_survives_preset():
    cfg = Config.from_dict({"table": {"preset": "pool-7ft", "length_in": 81.0}})
    assert cfg.table.length_in == pytest.approx(81.0)  # explicit value wins
    assert cfg.table.width_in == pytest.approx(39.0)  # preset fills the rest


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="lower_bound"):
        Config.from_dict({"detector": {"lower_bound": [30, 20, 200]}})
    with pytest.raises(ValueError, match="Unknown top-level"):
        Config.from_dict({"sense": 10})


def test_config_round_trip(tmp_path):
    cfg = Config()
    cfg.tracker.max_speed_in_s = 555.0
    path = tmp_path / "cfg.json"
    cfg.dump(path)
    back = Config.load(path)
    assert back.tracker.max_speed_in_s == pytest.approx(555.0)
    assert json.loads(path.read_text())["table"]["preset"] in TABLE_PRESETS


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def test_hungarian_matches_scipy_on_random_matrices():
    scipy_lsa = pytest.importorskip("scipy.optimize").linear_sum_assignment
    rng = np.random.default_rng(7)
    for _ in range(120):
        n, m = rng.integers(1, 8), rng.integers(1, 8)
        cost = rng.random((n, m)) * 100
        if rng.random() < 0.4:
            cost[rng.random(cost.shape) < 0.35] = FORBIDDEN
        r1, c1 = scipy_lsa(cost)
        r2, c2 = _hungarian(cost)
        assert len(r2) == min(n, m)
        assert cost[r1, c1].sum() == pytest.approx(cost[r2, c2].sum())


def test_solve_is_globally_optimal_not_greedy():
    # Greedy takes (0,0)=1 and is then forced into (1,1)=100 for a total of 101.
    # The optimal pairing is (0,1)+(1,0) = 2+3 = 5.
    cost = np.array([[1.0, 2.0], [3.0, 100.0]])
    rows, cols = solve(cost)
    assert cost[rows, cols].sum() == pytest.approx(5.0)


def test_associate_reports_unmatched():
    cost = np.array([[0.2, FORBIDDEN], [FORBIDDEN, FORBIDDEN]])
    matches, un_rows, un_cols = associate(cost)
    assert matches == [(0, 0)]
    assert un_rows == [1]
    assert un_cols == [1]


def test_associate_handles_empty():
    assert associate(np.zeros((0, 0))) == ([], [], [])


def test_the_numpy_fallback_is_used_when_scipy_is_absent(monkeypatch):
    """The README promises NumPy and OpenCV are the only hard requirements.

    Verify the fallback really is wired up, and that it returns the same
    assignment SciPy does on a matrix where a greedy answer would differ.
    """
    import billiards.assignment as assignment

    monkeypatch.setattr(assignment, "HAVE_SCIPY", False)
    cost = np.array([[1.0, 2.0, 9.0], [3.0, 100.0, 9.0], [9.0, 9.0, 0.5]])
    rows, cols = assignment.solve(cost)
    assert cost[rows, cols].sum() == pytest.approx(5.5)


def test_json_config_works_without_pyyaml(monkeypatch, tmp_path):
    import billiards.config as config_module

    monkeypatch.setattr(config_module, "yaml", None)
    cfg = Config()
    cfg.tracker.max_speed_in_s = 321.0

    path = tmp_path / "cfg.json"
    cfg.dump(path)
    assert Config.load(path).tracker.max_speed_in_s == pytest.approx(321.0)

    with pytest.raises(RuntimeError, match="PyYAML is required"):
        cfg.dump(tmp_path / "cfg.yaml")


# --------------------------------------------------------------------------
# Kalman
# --------------------------------------------------------------------------


def test_kalman_predicts_straight_line_without_friction():
    kf = BallKalman((0.0, 0.0), velocity_tau_s=1e6, initial_velocity=(60.0, 0.0))
    for _ in range(30):
        kf.predict(1 / 30)
    assert kf.position[0] == pytest.approx(60.0, rel=1e-3)
    assert kf.position[1] == pytest.approx(0.0, abs=1e-6)


def test_kalman_friction_slows_the_ball():
    kf = BallKalman((0.0, 0.0), velocity_tau_s=1.0, initial_velocity=(100.0, 0.0))
    kf.predict(1.0)
    # After one time constant the speed should be down by a factor of e.
    assert kf.speed == pytest.approx(100.0 / np.e, rel=1e-6)
    # And it travelled less far than a frictionless ball would have.
    assert kf.position[0] < 100.0


def test_kalman_converges_to_noisy_measurements():
    rng = np.random.default_rng(3)
    truth = np.array([10.0, 20.0])
    kf = BallKalman(tuple(truth + rng.normal(0, 0.5, 2)), meas_std_in=0.2)
    for _ in range(60):
        kf.predict(1 / 30)
        kf.update(tuple(truth + rng.normal(0, 0.2, 2)))
    assert np.linalg.norm(kf.position - truth) < 0.25
    assert kf.speed < 5.0  # a stationary ball must not accumulate velocity


def test_kalman_tracks_constant_velocity_measurements():
    kf = BallKalman((0.0, 25.0), meas_std_in=0.2)
    v = 40.0
    for i in range(1, 61):
        kf.predict(1 / 30)
        kf.update((v * i / 30.0, 25.0))
    assert kf.velocity[0] == pytest.approx(v, rel=0.12)


def test_apply_impulse_increases_velocity_uncertainty():
    kf = BallKalman((0.0, 0.0))
    before = kf.P[2, 2]
    kf.apply_impulse()
    assert kf.P[2, 2] > before


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_order_corners_is_rotation_invariant():
    quad = np.array([[10.0, 10.0], [100.0, 20.0], [95.0, 80.0], [5.0, 70.0]])
    expected = order_corners(quad)
    for shift in range(4):
        rolled = np.roll(quad, shift, axis=0)
        assert np.allclose(order_corners(rolled), expected)
    # Reversed winding must give the same answer too.
    assert np.allclose(order_corners(quad[::-1]), expected)


def _demo_table() -> TableModel:
    corners = np.array([[300.0, 180.0], [980.0, 180.0], [1180.0, 600.0], [100.0, 600.0]])
    return TableModel(corners, length_in=100.0, width_in=50.0, ball_diameter_in=2.25)


def test_homography_round_trip():
    table = _demo_table()
    pts = np.array([[10.0, 10.0], [50.0, 25.0], [90.0, 40.0]])
    back = table.image_to_table(table.table_to_image(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_corners_map_to_table_rectangle():
    table = _demo_table()
    mapped = table.image_to_table(table.corners_image)
    # Whichever way round the orientation test decides, the four corners must
    # land on the four corners of a 100 x 50 rectangle.
    assert np.allclose(sorted(map(tuple, np.round(mapped, 6))),
                       sorted([(0.0, 0.0), (0.0, 50.0), (100.0, 0.0), (100.0, 50.0)]),
                       atol=1e-6)


def _project_table(yaw_deg: float, height_in: float = 60.0,
                   distance_in: float = 95.0, focal_px: float = 1300.0,
                   image_size=(1280, 720)):
    """Project a real 100x50 table through a real pinhole camera.

    ``yaw_deg=0`` films the table from behind one end rail, so its 100-inch
    length runs *into* the frame and is heavily foreshortened.  ``yaw_deg=90``
    films it from the side, so the length runs across the frame.
    """
    import cv2

    L, W = 100.0, 50.0
    corners = np.array(
        [[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], dtype=np.float64
    )
    centre = np.array([L / 2.0, W / 2.0, 0.0])

    yaw = np.deg2rad(yaw_deg)
    # Camera sits `distance_in` from the table centre, `height_in` above it.
    eye = centre + np.array(
        [-distance_in * np.cos(yaw), -distance_in * np.sin(yaw), height_in]
    )
    forward = centre - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])  # world -> camera

    cam = (corners - eye) @ R.T
    K = np.array(
        [[focal_px, 0, image_size[0] / 2.0],
         [0, focal_px, image_size[1] / 2.0],
         [0, 0, 1.0]]
    )
    projected = cam @ K.T
    return projected[:, :2] / projected[:, 2:3], image_size


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 35.0])
def test_table_orientation_is_recovered_from_a_real_projection(yaw_deg):
    """The scale error that broke real footage.

    Filmed from behind an end rail, the 100-inch length of a pool table
    subtends *fewer* pixels than the 50-inch cushion nearest the camera, so the
    obvious "longer edge is the long side" rule picks the wrong axis and every
    distance comes out a factor of two off. The orientation has to be decided by
    which assignment a real camera could have produced, not by pixel counts.
    """
    from billiards.geometry import order_corners

    projected, image_size = _project_table(yaw_deg)
    table = TableModel(
        order_corners(projected), length_in=100.0, width_in=50.0,
        ball_diameter_in=2.25, image_size=image_size,
    )
    mapped = table.image_to_table(projected)

    # Corner 0 -> (0,0) and corner 1 -> (100,0): the 100-inch edge of the real
    # table must come back as a 100-inch edge.
    assert np.linalg.norm(mapped[0] - mapped[1]) == pytest.approx(100.0, abs=0.5)
    assert np.linalg.norm(mapped[1] - mapped[2]) == pytest.approx(50.0, abs=0.5)


def test_sphere_scale_exceeds_flat_scale_under_perspective():
    """A ball is a sphere, so it is not foreshortened the way a painted disc is.

    Using the flat-plane scale under-predicts ball size on exactly the angle
    pool is normally filmed from, which made every ball look like a multi-ball
    cluster and get split in half.
    """
    projected, image_size = _project_table(0.0)
    from billiards.geometry import order_corners

    table = TableModel(
        order_corners(projected), length_in=100.0, width_in=50.0,
        ball_diameter_in=2.25, image_size=image_size,
    )
    centre_img = tuple(table.table_to_image([(50.0, 25.0)])[0])
    assert table.sphere_scale_at(centre_img) > 1.3 * table.px_per_inch_at(centre_img)


def test_perspective_scale_varies_across_the_table():
    """Balls near the camera must be expected to be bigger than far ones.

    A single global ball-size threshold cannot hold on an angled view; this is
    the property that makes the size gate work anyway.
    """
    table = _demo_table()
    near = table.expected_ball_radius_px(tuple(np.mean(table.corners_image[2:4], axis=0)))
    far = table.expected_ball_radius_px(tuple(np.mean(table.corners_image[0:2], axis=0)))
    assert near > far * 1.2


def test_pockets_are_at_the_corners_and_middles():
    table = _demo_table()
    pockets = table.pockets_table()
    assert pockets.shape == (6, 2)
    assert table.nearest_pocket_distance((0.0, 0.0)) == pytest.approx(0.0)
    assert table.nearest_pocket_distance((50.0, 25.0)) == pytest.approx(25.0)


def test_carom_table_has_no_pockets():
    table = TableModel(
        _demo_table().corners_image, 111.8, 55.9, 2.42, has_pockets=False
    )
    assert table.pockets_table().shape == (0, 2)
    assert table.nearest_pocket_distance((10.0, 10.0)) == float("inf")


def test_quad_from_contour_recovers_a_trapezoid():
    import cv2

    quad = np.array([[300, 180], [980, 180], [1180, 600], [100, 600]], dtype=np.int32)
    img = np.zeros((720, 1280), dtype=np.uint8)
    cv2.fillConvexPoly(img, quad, 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = quad_from_contour(contours[0])
    assert found is not None
    assert np.max(np.abs(found - order_corners(quad.astype(float)))) < 3.0


def test_quad_survives_corners_bitten_off_by_pockets():
    """The real reason the old corner finder failed.

    Pockets remove the actual corner of the cloth.  Any method that picks a
    *contour point* as the corner is then wrong by the pocket radius; fitting
    the cushion lines and intersecting them is not.
    """
    import cv2

    quad = np.array([[300, 180], [980, 180], [1180, 600], [100, 600]], dtype=np.int32)
    img = np.zeros((720, 1280), dtype=np.uint8)
    cv2.fillConvexPoly(img, quad, 255)
    for corner in quad:
        cv2.circle(img, tuple(corner), 26, 0, -1)
    for mid in [((300 + 980) // 2, 180), ((100 + 1180) // 2, 600)]:
        cv2.circle(img, mid, 24, 0, -1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = quad_from_contour(max(contours, key=cv2.contourArea))
    assert found is not None
    assert np.max(np.abs(found - order_corners(quad.astype(float)))) < 4.0


# --------------------------------------------------------------------------
# Colour signatures
# --------------------------------------------------------------------------


def test_colour_distance_is_small_for_the_same_colour():
    a = ColorSignature(lab=np.array([150.0, 130.0, 190.0]), white_fraction=0.05)
    b = ColorSignature(lab=np.array([148.0, 131.0, 188.0]), white_fraction=0.06)
    c = ColorSignature(lab=np.array([150.0, 190.0, 110.0]), white_fraction=0.05)
    assert a.distance(b) < 5.0
    assert a.distance(c) > 40.0


def test_cue_ball_classification():
    cue = ColorSignature(
        lab=np.array([245.0, 128.0, 128.0]), white_fraction=0.95, chroma=4.0
    )
    eight = ColorSignature(
        lab=np.array([35.0, 128.0, 128.0]), white_fraction=0.0, chroma=3.0
    )
    stripe = ColorSignature(
        lab=np.array([170.0, 110.0, 200.0]), white_fraction=0.45, chroma=60.0
    )
    solid = ColorSignature(
        lab=np.array([120.0, 110.0, 200.0]), white_fraction=0.02, chroma=70.0
    )
    assert cue.classify() == "cue"
    assert eight.classify() == "eight"
    assert stripe.classify() == "stripe"
    assert solid.classify() == "solid"


def test_a_mostly_white_stripe_is_not_mistaken_for_the_cue_ball():
    """Stripes are white balls with one coloured band, so they are mostly white.

    Classifying on white area alone labelled every stripe "CUE".  The cue ball
    carries no colour anywhere; a stripe carries a lot in one band, which shows
    up in the high percentile of chroma but not in its median.
    """
    cue = ColorSignature(
        lab=np.array([243.0, 128.0, 128.0]),
        white_fraction=0.93, chroma=5.0, chroma_high=9.0,
    )
    stripe = ColorSignature(
        lab=np.array([210.0, 120.0, 190.0]),
        white_fraction=0.68, chroma=12.0, chroma_high=78.0,
    )
    assert cue.classify() == "cue"
    assert stripe.classify() == "stripe"


def test_signature_blend_moves_towards_target():
    a = ColorSignature(lab=np.array([100.0, 100.0, 100.0]), white_fraction=0.0)
    b = ColorSignature(lab=np.array([200.0, 200.0, 200.0]), white_fraction=1.0)
    mid = a.blend(b, 0.5)
    assert np.allclose(mid.lab, [150.0, 150.0, 150.0])
    assert mid.white_fraction == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Telling a ball from a hand
# --------------------------------------------------------------------------


def _lab_of(bgr, shape=(160, 160)):
    """A solid-colour Lab image to draw test objects on."""
    img = np.zeros((shape[0], shape[1], 3), np.uint8)
    img[:] = bgr
    return img


def test_rim_contrast_is_high_for_a_ball_on_cloth():
    """A ball ends at its rim: one radius out, the colour is cloth."""
    import cv2

    from billiards.detect import rim_contrast, sample_signature

    frame = _lab_of((120, 130, 90))  # cloth
    cv2.circle(frame, (80, 80), 14, (40, 40, 220), -1)  # a red ball
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    sig = sample_signature(lab, (80.0, 80.0), 14.0)
    assert rim_contrast(lab, (80.0, 80.0), 14.0, sig) > 30.0


def test_rim_contrast_is_low_for_a_disc_inside_something_larger():
    """The hand case.  A ball-sized, ball-round, ball-thick patch of a hand
    passes every shape test there is; what it cannot do is stop being a hand
    one radius further out."""
    import cv2

    from billiards.detect import rim_contrast, sample_signature

    frame = _lab_of((120, 130, 90))
    cv2.circle(frame, (80, 80), 55, (90, 110, 160), -1)  # a big patch of skin
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    sig = sample_signature(lab, (80.0, 80.0), 14.0)
    assert rim_contrast(lab, (80.0, 80.0), 14.0, sig) < 5.0


def test_rim_contrast_survives_a_ball_touching_a_neighbour():
    """Most of the rim still meets cloth, and the median only needs most."""
    import cv2

    from billiards.detect import rim_contrast, sample_signature

    frame = _lab_of((120, 130, 90))
    cv2.circle(frame, (80, 80), 14, (40, 40, 220), -1)
    cv2.circle(frame, (108, 80), 14, (220, 60, 40), -1)  # touching neighbour
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    sig = sample_signature(lab, (80.0, 80.0), 14.0)
    assert rim_contrast(lab, (80.0, 80.0), 14.0, sig) > 25.0


def _detector_on(frame, cfg=None):
    """A BallDetector looking at a plain overhead table filling the frame."""
    import cv2

    from billiards.detect import BallDetector
    from billiards.table import ClothModel

    cfg = cfg or Config()
    h, w = frame.shape[:2]
    corners = np.array(
        [[10.0, 10.0], [w - 10.0, 10.0], [w - 10.0, h - 10.0], [10.0, h - 10.0]]
    )
    table = TableModel(
        corners,
        length_in=cfg.table.length_in,
        width_in=cfg.table.width_in,
        ball_diameter_in=cfg.table.ball_diameter_in,
        has_pockets=False,
        image_size=(w, h),
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sample = hsv[5, 5]
    cloth = ClothModel(
        hue=float(sample[0]), sat=float(sample[1]), val=float(sample[2]),
        hue_halfwidth=8.0, sat_halfwidth=45.0, val_halfwidth=60.0,
    )
    return BallDetector(cfg, table, cloth), table


def _blank_table(w=800, h=400, cloth=(120, 130, 90)):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = cloth
    return frame


def test_a_cluster_of_touching_balls_is_still_detected():
    """The guard below must not cost us a rack."""
    import cv2

    frame = _blank_table()
    detector, table = _detector_on(frame)
    r = int(round(table.expected_ball_radius_px((400.0, 200.0))))
    centres = [(400, 200), (400 + 2 * r, 200), (400 + r, 200 + 2 * r)]
    for i, (x, y) in enumerate(centres):
        cv2.circle(frame, (x, y), r, [(40, 40, 220), (220, 60, 40), (30, 200, 230)][i], -1)

    found = detector.detect(frame)
    assert len(found) >= 3, [d.to_dict() for d in found]


def test_a_hand_shaped_blob_yields_no_balls():
    """A palm with fingers: ball-thick, ball-round in places, and ten times a
    ball in area.  Splitting it finds convincing round peaks; the discs those
    peaks imply account for almost none of it, which is the tell."""
    import cv2

    frame = _blank_table()
    detector, table = _detector_on(frame)
    r = int(round(table.expected_ball_radius_px((400.0, 200.0))))
    skin = (90, 110, 160)
    cv2.ellipse(frame, (400, 200), (3 * r, 2 * r), 0, 0, 360, skin, -1)
    for dx in (-2, -1, 0, 1, 2):
        cv2.line(
            frame, (400 + dx * r, 200), (400 + dx * r, 200 - 5 * r), skin, max(2, r // 2)
        )

    found = detector.detect(frame)
    assert found == [], [d.to_dict() for d in found]
    assert detector.last_debug["rejected"]["not_balls"] >= 1


def test_a_rack_the_splitter_under_separates_is_still_believed():
    """Failing to account for a blob does not prove it is not balls.

    Same-coloured neighbours in a rack share no visible edge, so the splitter
    finds five of eight and the discs cover 0.60 of the blob.  Rejecting on
    that alone cost nine points of recall against ground truth.  The balls it
    did find still sit on unmistakable circular edges, which is the second way
    a blob can be believed.
    """
    import cv2

    frame = _blank_table()
    detector, table = _detector_on(frame)
    r = int(round(table.expected_ball_radius_px((400.0, 200.0))))
    # Nine balls racked tight, three of them sharing a colour with a neighbour
    # so that the splitter cannot separate every pair.
    colours = [
        (40, 40, 220), (40, 40, 220), (220, 60, 40),
        (30, 200, 230), (30, 200, 230), (140, 50, 110),
        (60, 140, 55), (60, 140, 55), (24, 24, 26),
    ]
    k = 0
    for row in range(3):
        for col in range(row + 1):
            x = 400 + row * int(1.74 * r)
            y = 200 + (2 * col - row) * r
            cv2.circle(frame, (x, y), r, colours[k], -1)
            k += 1

    found = detector.detect(frame)
    assert len(found) >= 4, [d.to_dict() for d in found]


def test_the_cluster_guard_can_be_switched_off():
    cfg = Config()
    cfg.detector.cluster_core_coverage_min = 0.0
    cfg.detector.rim_contrast_min = 0.0
    import cv2

    frame = _blank_table()
    detector, table = _detector_on(frame, cfg)
    r = int(round(table.expected_ball_radius_px((400.0, 200.0))))
    cv2.ellipse(frame, (400, 200), (3 * r, 2 * r), 0, 0, 360, (90, 110, 160), -1)
    assert detector.detect(frame), "guard disabled, so the blob should split"


# --------------------------------------------------------------------------
# Coasting is extrapolation, and has to be paid for
# --------------------------------------------------------------------------


def _tracker_with(cfg=None):
    from billiards.track import MultiObjectTracker

    cfg = cfg or Config()
    return MultiObjectTracker(cfg, _demo_table(), fps=30.0), cfg


def _detection_at(xy, cfg):
    from billiards.detect import ColorSignature, Detection

    return Detection(
        centre_image=(0.0, 0.0),
        centre_table=xy,
        radius_px=10.0,
        area_ratio=1.0,
        circularity=0.9,
        signature=ColorSignature(lab=np.array([150.0, 130.0, 130.0])),
        rim_contrast=40.0,
    )


def test_a_barely_seen_track_is_dropped_long_before_a_well_seen_one():
    """A blob that looked like a ball three frames running should not then
    draw forty-five frames of confident trajectory on the strength of it."""
    tracker, cfg = _tracker_with()
    pos = (50.0, 25.0)

    for _ in range(cfg.tracker.min_hits_to_confirm):
        tracker.update([_detection_at(pos, cfg)], 1 / 30, 0, 0.0)
    flimsy = tracker.tracks[0]
    assert flimsy.state.value == "confirmed"

    for i in range(20):
        tracker.update([], 1 / 30, i, i / 30.0)
    assert flimsy.death_reason == "lost"
    assert flimsy.death_frame is not None and flimsy.death_frame <= 5


def test_a_long_lived_track_still_gets_the_full_coasting_window():
    tracker, cfg = _tracker_with()
    pos = (50.0, 25.0)
    for i in range(120):
        tracker.update([_detection_at(pos, cfg)], 1 / 30, i, i / 30.0)
    solid = tracker.tracks[0]

    for i in range(cfg.tracker.max_age_coasting):
        tracker.update([], 1 / 30, 200 + i, (200 + i) / 30.0)
    assert solid.is_alive, "a ball watched for 120 frames must survive an occlusion"



# --------------------------------------------------------------------------
# Seeing a camera cut
# --------------------------------------------------------------------------


def _pipeline_over(frame):
    from billiards.pipeline import TrackingPipeline
    from billiards.table import ClothModel
    import cv2

    h, w = frame.shape[:2]
    corners = np.array(
        [[10.0, 10.0], [w - 10.0, 10.0], [w - 10.0, h - 10.0], [10.0, h - 10.0]]
    )
    cfg = Config()
    table = TableModel(
        corners,
        length_in=cfg.table.length_in,
        width_in=cfg.table.width_in,
        ball_diameter_in=cfg.table.ball_diameter_in,
        has_pockets=False,
        image_size=(w, h),
    )
    sample = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[5, 5]
    cloth = ClothModel(
        hue=float(sample[0]), sat=float(sample[1]), val=float(sample[2]),
        hue_halfwidth=8.0, sat_halfwidth=45.0, val_halfwidth=60.0,
    )
    return TrackingPipeline(cfg, table, cloth, fps=30.0), cfg


def test_a_ball_rolling_across_the_bed_does_not_read_as_a_cut():
    """The margin this signal lives or dies by.

    Measured over the three sample clips, the busiest frame of play repaints
    6% of the bed and a typical one under 2%, against 16-32% for a cut.  A
    ball is a thousandth of the bed, so even nine of them moving at once come
    nowhere near the threshold.
    """
    import cv2

    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    r = int(round(pipe.table.expected_ball_radius_px((400.0, 200.0))))

    before = frame.copy()
    for i in range(9):
        cv2.circle(before, (100 + 70 * i, 200), r, (40, 40, 220), -1)
    after = frame.copy()
    for i in range(9):
        cv2.circle(after, (100 + 70 * i, 260), r, (40, 40, 220), -1)

    pipe._bed_repainted(before)
    assert not pipe._bed_repainted(after)


def test_a_cut_is_seen_the_frame_it_starts():
    """No patience, because a cut is not ambiguous.  Waiting for cloth coverage
    to collapse instead cost twelve frames on real footage, which was long
    enough to confirm thirty phantom tracks."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)

    rng = np.random.default_rng(1)
    elsewhere = rng.integers(0, 256, frame.shape, dtype=np.uint8)

    pipe._bed_repainted(frame)
    assert pipe._bed_repainted(elsewhere)


def test_the_cut_test_can_be_switched_off():
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    pipe.cfg.table.view_change_area_ratio = 0.0
    rng = np.random.default_rng(1)
    pipe._bed_repainted(frame)
    assert not pipe._bed_repainted(
        rng.integers(0, 256, frame.shape, dtype=np.uint8)
    )


# --------------------------------------------------------------------------
# Seeing a repeated frame
# --------------------------------------------------------------------------


def _with_ball_at(frame, pipe, xy):
    import cv2

    out = frame.copy()
    r = int(round(pipe.table.expected_ball_radius_px(xy)))
    cv2.circle(out, (int(xy[0]), int(xy[1])), r, (40, 40, 220), -1)
    return out


def test_an_identical_frame_is_recognised_as_a_repeat():
    """Broadcast clips are routinely 25 fps content rewrapped at 37.7, so one
    frame in three is a copy.  Measuring a copy again tells the filter the ball
    did not move, which is not what the picture says -- it says nothing."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    first = _with_ball_at(frame, pipe, (400.0, 200.0))

    pipe._bed_change(first)
    assert pipe._bed_change(first.copy())[1], "a pixel-identical frame is a repeat"


def test_a_re_encoded_copy_is_still_a_repeat():
    """A copy in a re-encoded stream is not bit-identical, which is why the
    test is counted at a third of full scale rather than at zero.  Measured
    over the three sample clips, a copy moves no bed pixel that far."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    first = _with_ball_at(frame, pipe, (400.0, 200.0))

    rng = np.random.default_rng(7)
    noise = rng.integers(-6, 7, first.shape, dtype=np.int16)
    grainy = np.clip(first.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pipe._bed_change(first)
    assert pipe._bed_change(grainy)[1]


def test_a_frame_with_a_ball_that_moved_is_not_a_repeat():
    """The gate that matters: it must not swallow real motion.  A ball rolling
    at 20 in/s moves about a fifth of its own width per frame."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    r = pipe.table.expected_ball_radius_px((400.0, 200.0))

    pipe._bed_change(_with_ball_at(frame, pipe, (400.0, 200.0)))
    moved = _with_ball_at(frame, pipe, (400.0 + 0.4 * r, 200.0))
    assert not pipe._bed_change(moved)[1]


def test_a_cut_is_never_read_as_a_repeat():
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    rng = np.random.default_rng(3)
    elsewhere = rng.integers(0, 256, frame.shape, dtype=np.uint8)

    pipe._bed_change(frame)
    repainted, repeats = pipe._bed_change(elsewhere)
    assert repainted and not repeats


def test_a_repeated_frame_is_replayed_rather_than_measured_again():
    """The point of the whole thing: the tracker's clock advances by the
    interval the picture actually changed across, not by the number of copies
    the file happened to contain."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    first = _with_ball_at(frame, pipe, (400.0, 200.0))

    pipe.process(first, 0, annotate=False)
    measured = pipe.last_frame_index

    result = pipe.process(first.copy(), 1, annotate=False)
    assert pipe.frames_repeated == 1
    assert pipe.last_frame_index == measured, "a copy must not advance the clock"
    # The row is still reported, so a per-frame export stays one row per ball
    # per frame and the annotated video stays one frame per input frame.
    assert result.frame_index == 1
    assert result.events == [], "nothing happened on a frame that did not change"


def test_a_run_of_repeats_is_capped():
    """The bed is genuinely still between shots -- two seconds on the sample
    clips -- and extrapolating a Kalman filter across that in one step makes
    its association gate wider than the table."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    still = _with_ball_at(frame, pipe, (400.0, 200.0))

    pipe.process(still, 0, annotate=False)
    for i in range(1, 2 + 2 * cfg.table.repeat_frame_max_run):
        pipe.process(still.copy(), i, annotate=False)

    assert pipe.frames_repeated < i, "a still table must still be measured now and then"
    assert pipe.last_frame_index > cfg.table.repeat_frame_max_run


def test_the_repeat_test_can_be_switched_off():
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    pipe.cfg.table.repeat_frame_max_run = 0
    still = _with_ball_at(frame, pipe, (400.0, 200.0))

    pipe.process(still, 0, annotate=False)
    pipe.process(still.copy(), 1, annotate=False)
    assert pipe.frames_repeated == 0
    assert pipe.last_frame_index == 1


# --------------------------------------------------------------------------
# Where the overhead diagram goes
# --------------------------------------------------------------------------


def test_the_overhead_diagram_does_not_cover_the_picture():
    """A pool camera fills its frame with table -- on the sample clips the bed
    covers the whole lower half -- so an inset in any corner sits on top of the
    thing the viewer is looking at.  It goes in a bar underneath instead."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)

    out = pipe.renderer.draw(frame, [], [], [], hud={"frame": "0"})
    h, w = frame.shape[:2]
    assert out.shape[1] == w, "the picture keeps its width"
    assert out.shape[0] > h, "and gains a bar below it"

    # Turning the diagram on must not change a single pixel of the picture.
    pipe.cfg.render.overhead_panel = False
    without = pipe.renderer.draw(frame, [], [], [], hud={"frame": "0"})
    assert np.array_equal(out[:h], without[:h])


def test_every_frame_of_the_output_is_the_same_size():
    """Including the ones where the table is off screen.  The writer adopts the
    first frame's size and squashes the rest, so a paused frame that skipped
    the bar would silently distort the whole clip from that point."""
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)

    tracked = pipe.renderer.draw(frame, [], [], [], hud={"frame": "0"})
    idle = pipe.renderer.compose_idle(frame, {"status": "paused"})
    assert tracked.shape == idle.shape


def test_the_diagram_can_still_be_an_inset():
    frame = _blank_table()
    pipe, cfg = _pipeline_over(frame)
    pipe.cfg.render.overhead_panel_place = "inset"

    out = pipe.renderer.draw(frame, [], [], [], hud={"frame": "0"})
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame), "the inset is drawn over the picture"
