"""Ball numbers, and the colour measurements they are read from."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from billiards import balls
from billiards.detect import colour_distance_matrix, sample_signature


# --------------------------------------------------------------------------
# Which numbers can be on the table
# --------------------------------------------------------------------------


def test_numbers_are_parsed_from_ranges_and_lists():
    assert balls.parse_numbers("1-9") == list(range(1, 10))
    assert balls.parse_numbers("1-7, 9") == [1, 2, 3, 4, 5, 6, 7, 9]
    assert balls.parse_numbers([3, 1, 3]) == [1, 3]


def test_the_8_is_never_a_numbered_colour():
    """The 8 is the 8-ball role's; it is not matched by colour."""
    numbers = [s.number for s in balls.ball_specs("standard", range(1, 16))]
    assert 8 not in numbers and len(numbers) == 14


def test_the_sets_differ_where_they_should():
    std = {s.number: s.family for s in balls.ball_specs("standard", range(1, 16))}
    tv = {s.number: s.family for s in balls.ball_specs("tv", range(1, 16))}
    assert (std[4], std[5]) == ("purple", "orange")
    assert (tv[4], tv[5]) == ("pink", "purple")
    assert std[12] == "purple" and tv[12] == "pink"


# --------------------------------------------------------------------------
# Naming a table's balls
# --------------------------------------------------------------------------


def _observed(family: str, stripe: float = 0.1, dark: float = 0.0, scale: float = 1.0):
    L, hue, C = balls.FAMILIES[family]
    return (L * scale, hue, C * scale, stripe, dark)


#: fedor_shot, as the tracker measured it: 2 3 4 6 7 9 (plus cue and 8).
FEDOR_SHOT = [(66, 272, 26, 0.05, 0.13), (95, 6, 50, 0.27, 0.0), (140, 5, 36, 0.20, 0.0),
              (120, 174, 32, 0.12, 0.02), (98, 31, 26, 0.08, 0.02), (159, 66, 55, 0.45, 0.12)]


def test_a_real_table_is_named_correctly():
    specs = balls.ball_specs("tv", range(1, 16))
    numbers, cost, _ = balls.assign(FEDOR_SHOT, specs, max_cost=3.0)
    assert numbers == [2, 3, 4, 6, 7, 9]
    assert cost < 1.5


def test_no_number_is_given_twice():
    specs = balls.ball_specs("tv", range(1, 16))
    two_reds = [_observed("red"), _observed("red"), _observed("green")]
    numbers, _, _ = balls.assign(two_reds, specs, max_cost=3.0)
    assert len({n for n in numbers if n is not None}) == len([n for n in numbers if n is not None])
    assert 3 in numbers and 6 in numbers


def test_a_colour_no_ball_has_stays_unnumbered():
    """A hand, a shaft, a glove: better no number than a wrong one."""
    specs = balls.ball_specs("tv", range(1, 16))
    grey = [(102.0, 297.0, 4.0, 0.0, 0.24)]  # albin_fedor's cue-shaft phantom
    numbers, _, _ = balls.assign(grey, specs, max_cost=3.0)
    assert numbers == [None]


def test_a_stripe_is_told_from_its_solid():
    specs = balls.ball_specs("standard", range(1, 16))
    pair = [_observed("blue", stripe=0.08), _observed("blue", stripe=0.5)]
    numbers, _, _ = balls.assign(pair, specs, max_cost=3.0)
    assert numbers == [2, 10]


def test_a_dim_venue_is_learned_from_the_table():
    """Every ball 30% darker and duller than the palette: the fitted scales
    absorb it and the names stay right."""
    specs = balls.ball_specs("tv", range(1, 10))
    dim = [(L * 0.7, hue, C * 0.7, s, d) for L, hue, C, s, d in FEDOR_SHOT]
    numbers, _, (kL, kC) = balls.assign(dim, specs, max_cost=3.0)
    assert numbers == [2, 3, 4, 6, 7, 9]
    assert kL == pytest.approx(0.7, abs=0.08) and kC == pytest.approx(0.7, abs=0.15)


def test_stickiness_keeps_a_close_call_where_it_was():
    specs = balls.ball_specs("tv", range(1, 16))
    # Between red and maroon in hue, closer to maroon.
    between = [(95.0, 22.0, 34.0, 0.1, 0.0)]
    fresh, _, _ = balls.assign(between, specs, max_cost=3.0)
    held, _, _ = balls.assign(between, specs, max_cost=3.0, current=[3], stickiness=1.0)
    assert fresh == [7] and held == [3]


def test_black_caps_say_tv_set():
    """With no pink and no orange ball on the table, a stripe's caps are what
    tells the sets apart: albin_fedor's 9 is black-capped."""
    nine = [(188.0, 89.0, 46.0, 0.66, 0.36)]
    tv = balls.cost_matrix(nine, balls.ball_specs("tv", [9]))[0, 0]
    std = balls.cost_matrix(nine, balls.ball_specs("standard", [9]))[0, 0]
    assert tv < std - 2.0


# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------


CLOTH = (176, 161, 147)  # the sample clips' blue-grey
IVORY = (166, 184, 190)


def _ball_image(colour, stripe_caps=None, number_circle=False, r=16):
    img = np.zeros((80, 80, 3), np.uint8)
    img[:] = CLOTH
    if stripe_caps is not None:
        cv2.circle(img, (40, 40), r, stripe_caps, -1)
        cv2.ellipse(img, (40, 40), (r, int(r * 0.55)), 0, 0, 360, colour, -1)
    else:
        cv2.circle(img, (40, 40), r, colour, -1)
    if number_circle:
        cv2.circle(img, (40, 40), int(r * 0.4), IVORY, -1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    cloth = np.zeros(img.shape[:2], np.uint8)
    cloth[np.all(img == CLOTH, axis=2)] = 255
    return sample_signature(lab, (40.0, 40.0), float(r), cloth)


RED = (85, 44, 162)


def test_the_tint_is_the_ball_colour_not_its_number_circle():
    plain = _ball_image(RED)
    facing = _ball_image(RED, number_circle=True)
    assert np.linalg.norm(plain.tint - facing.tint) < 4.0
    # ...whereas the white fraction the old colour distance leaned on jumps.
    assert facing.white_fraction - plain.white_fraction > 0.25


def test_a_number_circle_does_not_make_a_ball_look_like_another():
    """The failure that cost the tracker its identities on camera-realistic
    balls: a solid with its number circle facing the camera read as a
    different ball, and was refused by its own track."""
    plain, facing = _ball_image(RED), _ball_image(RED, number_circle=True)
    maroon = _ball_image((62, 68, 120))
    d = colour_distance_matrix([plain], [facing, maroon])[0]
    assert d[0] < 12.0 and d[1] > 2 * d[0]


@pytest.mark.parametrize("caps", [IVORY, (30, 30, 32)], ids=["white caps", "black caps"])
def test_a_stripe_scores_high_whatever_its_caps(caps):
    stripe = _ball_image(RED, stripe_caps=caps)
    solid = _ball_image(RED, number_circle=True)
    assert stripe.stripe > 0.35 > solid.stripe


def test_the_8_is_black_and_the_cue_ball_is_not():
    eight = _ball_image((24, 24, 26), number_circle=True)
    cue = _ball_image(IVORY)
    assert eight.dark_fraction > 0.5 and cue.dark_fraction < 0.05
    assert eight.eight_score > cue.eight_score
    assert cue.cue_score > eight.cue_score


def test_the_yellow_1_is_not_the_cue_ball():
    """Its white number circle made it read half white, and it took the cue
    ball's role once the cue ball had gone."""
    yellow = _ball_image((78, 140, 194), number_circle=True)
    cue = _ball_image(IVORY)
    assert cue.cue_score > yellow.cue_score + 0.5
    L, C = yellow.lightness_chroma
    assert C > 30.0


def test_a_colourless_ball_has_a_stable_colour():
    """A few stray coloured pixels at the 8's rim used to become its "colour",
    which jumped by 80-130 between frames."""
    a = _ball_image((24, 24, 26))
    b = _ball_image((24, 24, 26), number_circle=True)
    assert colour_distance_matrix([a], [b])[0, 0] < 20.0


def test_a_touching_neighbour_is_not_read_as_a_cap():
    """Sampled only on its own side of the line to the neighbour, a solid
    frozen against another ball still reads as a solid."""
    img = np.zeros((80, 110, 3), np.uint8)
    img[:] = CLOTH
    cv2.circle(img, (40, 40), 16, RED, -1)
    cv2.circle(img, (66, 40), 16, (99, 63, 26), -1)  # blue, overlapping in view
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    cloth = np.zeros(img.shape[:2], np.uint8)
    cloth[np.all(img == CLOTH, axis=2)] = 255
    alone = sample_signature(lab, (40.0, 40.0), 16.0, cloth)
    split = sample_signature(lab, (40.0, 40.0), 16.0, cloth, neighbours=[(66.0, 40.0)])
    assert split.stripe < alone.stripe
    assert split.stripe < 0.22
