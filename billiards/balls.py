"""Which numbered ball each track is.

The tracker knows every ball by an id it made up (``#5``), and the broadcast
knows them by the number printed on them.  The colours give the numbers away --
brown is the 7, green the 6 -- so this module turns one into the other.

Two things make it a question about the whole table rather than about each ball
on its own, exactly as the cue-ball and 8-ball roles are (``track.assign_roles``):

* there is at most one of each number, so a ball is the 3 partly because no
  other ball on the table is a better 3;
* colours on camera depend on the venue.  The sample broadcasts are far less
  saturated than a ball set looks in a catalogue, so a palette is compared
  after fitting one lightness scale and one saturation scale to the whole
  table: the *arrangement* of the colours is what identifies them.

The palette was measured on the three sample broadcasts: the median colour of
each ball's coloured pixels (``ColorSignature.tint``), OpenCV Lab.  Their set is
a TV set -- the 4 pink, the 5 purple, and stripes with black caps -- and a
standard set's 4 is purple and its 5 orange.  Orange appears on no sample clip,
so its entry is an estimate between their red and yellow.  Balls 9-15 share
the colours of 1-7 and are told apart by ``ColorSignature.stripe``, which reads
white caps and black caps alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .assignment import solve


#: (lightness, hue in degrees, chroma) of each colour of ball, in OpenCV's
#: 8-bit Lab (lightness 0-255; hue and chroma from a*, b* offset by 128),
#: as the sample broadcasts show them.
FAMILIES: Dict[str, Tuple[float, float, float]] = {
    "yellow": (160.0, 71.0, 44.0),
    "blue": (66.0, 271.0, 25.0),
    "red": (97.0, 4.0, 52.0),
    "pink": (140.0, 5.0, 36.0),
    "purple": (87.0, 306.0, 17.0),
    "green": (125.0, 173.0, 31.0),
    "maroon": (90.0, 30.0, 26.0),
    # Not on any sample clip: between their red and yellow.
    "orange": (128.0, 45.0, 50.0),
}

#: The colour of balls 1-7 in each set; 9-15 are the same colours, striped.
BALL_SETS: Dict[str, Dict[int, str]] = {
    "standard": {1: "yellow", 2: "blue", 3: "red", 4: "purple", 5: "orange",
                 6: "green", 7: "maroon"},
    "tv": {1: "yellow", 2: "blue", 3: "red", 4: "pink", 5: "purple",
           6: "green", 7: "maroon"},
}

#: Stripe score (``ColorSignature.stripe``) up to which a ball reads as a solid,
#: and from which as a stripe; each ``_STRIPE_SOFTNESS`` past its side costs
#: one unit.  A solid's number circle and highlight read 0.05-0.28 on the
#: sample clips and 0.03-0.33 on the synthetic ones; a stripe's caps 0.38-0.69
#: and 0.34-0.59.
_SOLID_UP_TO = 0.22
_STRIPE_FROM = 0.40
_STRIPE_SOFTNESS = 0.07

#: What colour a stripe's caps are in each set, read from the black fraction
#: (``ColorSignature.dark_fraction``): the sample clips' 9 reads 0.09-0.36, a
#: white-capped synthetic stripe 0.00.  It is what tells the two sets apart when
#: neither a pink nor an orange ball is on the table.
_CAPS_BLACK_FROM = 0.06
_CAPS_WHITE_UP_TO = 0.04
_CAPS_SOFTNESS = 0.03
CAPS: Dict[str, str] = {"standard": "white", "tv": "black"}

#: One unit of cost is this far off in each quantity.
_HUE_DEG = 22.0
_LOG_LIGHTNESS = 0.22
_LOG_CHROMA = 0.40
#: Below this chroma a colour has no reliable hue.
_HUE_CONFIDENT_CHROMA = 12.0

#: How far the table-wide lightness and saturation scales may stray from the
#: sample broadcasts'.
_SCALE_LIMITS = {"lightness": (0.6, 1.6), "chroma": (0.35, 2.8)}


def parse_numbers(spec: object) -> List[int]:
    """``"1-9"``, ``"1-7,9"`` or a list of ints -> the numbers allowed on the table."""
    if isinstance(spec, (list, tuple)):
        return sorted({int(n) for n in spec})
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


@dataclass(frozen=True)
class BallSpec:
    number: int
    family: str
    striped: bool
    caps: str = "white"

    @property
    def colour(self) -> Tuple[float, float, float]:
        return FAMILIES[self.family]


def ball_specs(ball_set: str, numbers: Iterable[int]) -> List[BallSpec]:
    """The numbered balls a set has, among ``numbers`` -- never the 8."""
    base = BALL_SETS[ball_set]
    specs = []
    for n in numbers:
        if n == 8 or not 1 <= n <= 15:
            continue
        solid = n if n < 8 else n - 8
        specs.append(BallSpec(n, base[solid], striped=n > 8, caps=CAPS[ball_set]))
    return specs


def colour_terms(tint: np.ndarray) -> Tuple[float, float, float]:
    """(lightness, hue in degrees, chroma) of an OpenCV Lab colour."""
    a, b = float(tint[1]) - 128.0, float(tint[2]) - 128.0
    return float(tint[0]), float(np.degrees(np.arctan2(b, a)) % 360.0), float(np.hypot(a, b))


def cost_matrix(
    observed: Sequence[Tuple[float, float, float, float]],
    specs: Sequence[BallSpec],
    lightness_scale: float = 1.0,
    chroma_scale: float = 1.0,
) -> np.ndarray:
    """How unlike each ball each observed colour is.

    ``observed`` is (lightness, hue, chroma, stripe score, black fraction) per
    track.  About 1 per quantity that is one typical deviation off; the total
    is the root sum of squares, so a colour right in hue but twice as
    saturated costs about 2.
    """
    if not len(observed) or not len(specs):
        return np.zeros((len(observed), len(specs)), dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1, 5)
    L, hue, C, stripe, dark = (obs[:, k:k + 1] for k in range(5))
    ref = np.array([s.colour for s in specs], dtype=np.float64)
    Le, he, Ce = ref[None, :, 0], ref[None, :, 1], ref[None, :, 2]
    striped = np.array([s.striped for s in specs])[None, :]
    black_caps = np.array([s.caps == "black" for s in specs])[None, :]

    hue_weight = np.minimum(1.0, C / _HUE_CONFIDENT_CHROMA)
    dh = np.abs((hue - he + 180.0) % 360.0 - 180.0) / _HUE_DEG * hue_weight
    dL = np.log(np.maximum(L, 1.0) / (lightness_scale * Le)) / _LOG_LIGHTNESS
    dC = np.log(np.maximum(C, 1.0) / (chroma_scale * Ce)) / _LOG_CHROMA
    ds = np.where(striped, np.maximum(0.0, _STRIPE_FROM - stripe),
                  np.maximum(0.0, stripe - _SOLID_UP_TO)) / _STRIPE_SOFTNESS
    caps = np.where(black_caps, np.maximum(0.0, _CAPS_BLACK_FROM - dark),
                    np.maximum(0.0, dark - _CAPS_WHITE_UP_TO)) / _CAPS_SOFTNESS
    dk = np.where(striped, caps, 0.0)
    return np.sqrt(dh * dh + dL * dL + dC * dC + ds * ds + dk * dk)


def assign(
    observed: Sequence[Tuple[float, float, float, float]],
    specs: Sequence[BallSpec],
    max_cost: float,
    current: Optional[Sequence[Optional[int]]] = None,
    stickiness: float = 0.0,
    lightness_scale: float = 1.0,
    chroma_scale: float = 1.0,
) -> Tuple[List[Optional[int]], float, Tuple[float, float]]:
    """Number every observed ball at once, at most one of each number.

    Returns (number per ball or None, mean cost of the numbered, the fitted
    (lightness, chroma) scales).  A ball whose best available number would
    cost more than ``max_cost`` gets none.  ``current`` numbers are favoured by
    ``stickiness``, so two similar balls do not trade numbers frame to frame.
    The two scales are refitted from the assignment and it is solved again,
    so a venue's exposure and saturation are learned from the table itself.
    """
    n = len(observed)
    if n == 0 or not specs:
        return [None] * n, 0.0, (lightness_scale, chroma_scale)
    numbers = [s.number for s in specs]
    kL, kC = lightness_scale, chroma_scale
    result: List[Optional[int]] = [None] * n
    mean_cost = 0.0
    for _ in range(2):
        cost = cost_matrix(observed, specs, kL, kC)
        if current is not None and stickiness > 0:
            for i, num in enumerate(current):
                if num in numbers:
                    cost[i, numbers.index(num)] -= stickiness
        # A "no number" column per ball, at the most a number may cost.
        padded = np.hstack([cost, np.full((n, n), max_cost)])
        rows, cols = solve(padded)
        result = [None] * n
        used = []
        for r, c in zip(rows, cols):
            if c < len(specs):
                result[r] = numbers[c]
                used.append((r, c))
        if not used:
            break
        raw = cost_matrix(observed, specs, kL, kC)
        mean_cost = float(np.mean([raw[r, c] for r, c in used]))
        # Refit the scales from what was assigned.
        ratios_L = [observed[r][0] / specs[c].colour[0] for r, c in used]
        ratios_C = [max(observed[r][2], 1.0) / specs[c].colour[2] for r, c in used]
        kL = float(np.clip(np.exp(np.median(np.log(ratios_L))), *_SCALE_LIMITS["lightness"]))
        kC = float(np.clip(np.exp(np.median(np.log(ratios_C))), *_SCALE_LIMITS["chroma"]))
    return result, mean_cost, (kL, kC)


#: The ball model's colour families (``billiards.ballnet.FAMILIES``), in its
#: order: every family of ``FAMILIES`` above, and black.
MODEL_FAMILIES = ["yellow", "blue", "red", "pink", "purple", "orange", "green", "maroon", "black"]

#: Probabilities from the ball model are clipped to this before their logs
#: are taken, so one confident mistake cannot make a number impossible.
_MODEL_FLOOR = 0.02


#: Colours one maker's ball can have where another's has the other: a set's
#: 7 is maroon, a light brown (the Aramith TV set) or orange (a ceiling
#: camera's club set, 28 Sep 2026).  Each counts for this much of the other.
_MODEL_NEIGHBOURS = {"maroon": "orange", "orange": "maroon"}
_MODEL_NEIGHBOUR_WEIGHT = 0.5
#: A stripe's caps, measured off the picture (``ColorSignature.dark_fraction``)
#: as the palette does, cost at most this much -- they are what tells the
#: two sets apart once the pink or orange ball is gone, and in 9-ball the 9
#: is on the table to the end.
_MODEL_CAPS_MAX = 1.5


def model_cost_matrix(
    family_logp: Sequence[np.ndarray],
    stripe_p: Sequence[float],
    specs: Sequence[BallSpec],
    dark: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """How unlike each ball each track is, by the ball model's evidence.

    ``family_logp`` is each track's average log probability of each colour
    family (``MODEL_FAMILIES``), ``stripe_p`` its average probability of being
    a stripe.  The cost of a number is minus the log probability of its
    family and of its pattern: 0.7 for an even chance of each, 2.3 for one
    in ten.  ``dark``, each track's black fraction, adds what a stripe's caps
    say.  In the same shape as ``cost_matrix``.
    """
    n, m = len(family_logp), len(specs)
    if not n or not m:
        return np.zeros((n, m), dtype=np.float64)
    fam = np.exp(np.asarray(family_logp, dtype=np.float64).reshape(n, len(MODEL_FAMILIES)))
    near = fam.copy()
    for a, b in _MODEL_NEIGHBOURS.items():
        near[:, MODEL_FAMILIES.index(a)] += _MODEL_NEIGHBOUR_WEIGHT * fam[:, MODEL_FAMILIES.index(b)]
    idx = np.array([MODEL_FAMILIES.index(s.family) for s in specs])
    striped = np.array([s.striped for s in specs])
    p = np.clip(np.asarray(stripe_p, dtype=np.float64).reshape(n, 1), _MODEL_FLOOR, 1.0 - _MODEL_FLOOR)
    pattern = np.where(striped[None, :], -np.log(p), -np.log(1.0 - p))
    cost = -np.log(np.clip(near[:, idx], _MODEL_FLOOR, None)) + pattern
    if dark is not None:
        d = np.asarray(dark, dtype=np.float64).reshape(n, 1)
        black_caps = np.array([s.caps == "black" for s in specs])[None, :]
        caps = np.where(black_caps, np.maximum(0.0, _CAPS_BLACK_FROM - d),
                        np.maximum(0.0, d - _CAPS_WHITE_UP_TO)) / _CAPS_SOFTNESS
        cost = cost + np.where(striped[None, :], np.minimum(caps, _MODEL_CAPS_MAX), 0.0)
    return cost


def assign_costs(
    cost: np.ndarray,
    specs: Sequence[BallSpec],
    max_cost: float,
    current: Optional[Sequence[Optional[int]]] = None,
    stickiness: float = 0.0,
) -> List[Optional[int]]:
    """Number every ball at once from a cost matrix, as ``assign`` does."""
    n = cost.shape[0]
    if n == 0 or not specs:
        return [None] * n
    numbers = [s.number for s in specs]
    cost = cost.copy()
    if current is not None and stickiness > 0:
        for i, num in enumerate(current):
            if num in numbers:
                cost[i, numbers.index(num)] -= stickiness
    padded = np.hstack([cost, np.full((n, n), max_cost)])
    rows, cols = solve(padded)
    result: List[Optional[int]] = [None] * n
    for r, c in zip(rows, cols):
        if c < len(specs):
            result[r] = numbers[c]
    return result
