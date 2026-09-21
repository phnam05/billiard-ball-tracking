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


def test_signature_blend_moves_towards_target():
    a = ColorSignature(lab=np.array([100.0, 100.0, 100.0]), white_fraction=0.0)
    b = ColorSignature(lab=np.array([200.0, 200.0, 200.0]), white_fraction=1.0)
    mid = a.blend(b, 0.5)
    assert np.allclose(mid.lab, [150.0, 150.0, 150.0])
    assert mid.white_fraction == pytest.approx(0.5)
