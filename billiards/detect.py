"""Ball detection without per-ball colour tuning.

The old detector asked "which pixels are inside this hand-picked HSV box?" and
took the single largest contour as the cue ball.  That needs one hand-tuned box
per ball per video, finds at most two balls, and merges into one blob whenever
two balls touch.

This detector asks the opposite question: **"what is on the table that is not
cloth?"**  The cloth colour is measured automatically (see ``table.py``), so
there is nothing left to tune by colour.  Everything else is decided by
geometry, using the expected ball size derived from the homography -- which is
computed *per image location*, so it stays correct under perspective where a
ball at the far cushion covers a quarter of the pixels of one at the near rail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import ballnet
from .config import Config
from .geometry import TableModel
from .table import ClothModel


# --------------------------------------------------------------------------
# Colour signature -- what makes a ball *that* ball
# --------------------------------------------------------------------------


@dataclass
class ColorSignature:
    """A ball's appearance, summarised so two views of it can be compared.

    CIE Lab is used rather than HSV because Euclidean distance in Lab is
    roughly perceptual, so a single distance threshold behaves the same for a
    yellow ball and a blue one.  Lightness is down-weighted in the distance
    because it is the channel that moves when the ball rolls through a shadow.
    """

    lab: np.ndarray  # (3,) float: L, a, b in OpenCV's 0..255 encoding
    white_fraction: float = 0.0
    chroma: float = 0.0
    #: High percentile of chroma across the disc.  A striped ball is mostly
    #: white with one strongly coloured band, so its *median* chroma is low
    #: while its high percentile is not; the cue ball is low in both.  Without
    #: this, every stripe is classified as the cue ball.
    chroma_high: float = 0.0
    #: The ball's own colour: the median of its *coloured* pixels only, out to
    #: near the rim and with cloth-coloured pixels left out.  ``lab`` is the
    #: median of everything in the inner disc, which a white number circle, a
    #: highlight or a stripe's caps pull towards grey; this is what says which
    #: ball it is.  None until measured; ``lab`` stands in for it.
    tint: Optional[np.ndarray] = None
    #: Fraction of the ball that is *not* its own colour -- caps, number
    #: circle, highlight.  A solid shows 0.07-0.21 of that on the sample clips
    #: and a stripe 0.43-0.64, whether its caps are white or, as on the TV
    #: ball set in those clips, black.  See ``billiards.balls``.
    stripe: float = 0.0
    #: Fraction of the ball that is black: nearly all of the 8, none of the
    #: cue ball.  Measured like ``stripe``.  It is what tells those two apart
    #: when the 8's white number circle faces the camera and fills the middle
    #: of the ball, which otherwise reads as half a cue ball.
    dark_fraction: float = 0.0

    @staticmethod
    def empty() -> "ColorSignature":
        return ColorSignature(lab=np.array([128.0, 128.0, 128.0]), white_fraction=0.0)

    @property
    def ball_colour(self) -> np.ndarray:
        """The ball's own colour in Lab: ``tint`` if measured, else ``lab``."""
        return self.lab if self.tint is None else self.tint

    def distance(self, other: "ColorSignature") -> float:
        """How different two views of a ball look.  See ``colour_distance_matrix``."""
        return float(colour_distance_matrix([self], [other])[0, 0])

    def blend(self, other: "ColorSignature", alpha: float) -> "ColorSignature":
        """Exponential moving average towards ``other``."""
        if self.tint is None or other.tint is None:
            tint = other.tint if self.tint is None else self.tint
        else:
            tint = (1.0 - alpha) * self.tint + alpha * other.tint
        return ColorSignature(
            lab=(1.0 - alpha) * self.lab + alpha * other.lab,
            white_fraction=(1.0 - alpha) * self.white_fraction
            + alpha * other.white_fraction,
            chroma=(1.0 - alpha) * self.chroma + alpha * other.chroma,
            chroma_high=(1.0 - alpha) * self.chroma_high + alpha * other.chroma_high,
            tint=tint,
            stripe=(1.0 - alpha) * self.stripe + alpha * other.stripe,
            dark_fraction=(1.0 - alpha) * self.dark_fraction + alpha * other.dark_fraction,
        )

    @property
    def bgr(self) -> Tuple[int, int, int]:
        """Approximate display colour for this signature: the ball's own."""
        patch = np.zeros((1, 1, 3), dtype=np.uint8)
        patch[0, 0] = np.clip(self.ball_colour, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(patch, cv2.COLOR_Lab2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    @property
    def lightness_chroma(self) -> Tuple[float, float]:
        """Lightness and chroma of the ball's own colour (``ball_colour``)."""
        c = self.ball_colour
        return float(c[0]), float(np.hypot(float(c[1]) - 128.0, float(c[2]) - 128.0))

    @property
    def cue_score(self) -> float:
        """How much this looks like *the* cue ball: bright, and colourless.

        A score rather than a test, because there is exactly one cue ball on the
        table and which ball it is, is a question about the whole set -- see
        MultiObjectTracker.assign_roles.  Judging each ball on its own gave two
        "cue balls" and four "8 balls" on a real clip.

        Read from the ball's own colour, not from how much of it is white: a
        solid's white number circle, facing the camera, made the yellow 1 read
        almost half white, and it took the cue ball's role once the cue ball
        was potted.  Measured, a cue ball's colour has a chroma of 5-13 and the
        1's 46-48.
        """
        L, C = self.lightness_chroma
        return L / 255.0 - C / 25.0

    @property
    def eight_score(self) -> float:
        """How much this looks like *the* 8 ball: black, and colourless."""
        L, C = self.lightness_chroma
        return max(self.dark_fraction, 1.0 - L / 128.0) - C / 25.0

    def classify(self) -> str:
        """Appearance class from this ball alone: ``cue``, ``stripe``,
        ``eight`` or ``solid``.

        The cue ball and a striped ball are both mostly white, so white area
        alone cannot separate them.  What does is that a stripe carries one
        strongly coloured band while the cue ball carries no colour anywhere:
        the median chroma of both is low, but the *high percentile* is not.
        """
        if self.white_fraction >= 0.6 and self.chroma_high < 30.0:
            return "cue"
        if self.lab[0] < 70.0 and self.chroma_high < 30.0:
            return "eight"
        if self.white_fraction >= 0.25:
            return "stripe"
        return "solid"

    def to_dict(self) -> dict:
        return {
            "lab": [round(float(x), 1) for x in self.lab],
            "tint": [round(float(x), 1) for x in self.ball_colour],
            "white_fraction": round(float(self.white_fraction), 3),
            "chroma": round(float(self.chroma), 1),
            "chroma_high": round(float(self.chroma_high), 1),
            "stripe": round(float(self.stripe), 3),
            "dark_fraction": round(float(self.dark_fraction), 3),
            "type": self.classify(),
        }


#: Per-channel weights used by ColorSignature.distance.  Lightness counts for
#: less because it is the channel that moves when a ball rolls through a shadow.
_LAB_WEIGHTS = np.array([0.45, 1.0, 1.0])
#: How much a difference in stripe score and in black fraction count, in the
#: same units as the colour difference.
_STRIPE_WEIGHT = 40.0
_DARK_WEIGHT = 60.0

#: The band round the table that should not be cloth (``surround_mask``), in
#: inches beyond the cushion noses: past the clothed cushion top, over the
#: wooden rail and onto the floor.
_SURROUND_INNER_IN = 5.0
_SURROUND_OUTER_IN = 11.0


def colour_distance_matrix(
    a: Sequence["ColorSignature"], b: Sequence["ColorSignature"]
) -> np.ndarray:
    """All pairwise colour distances at once.

    The ball's own colour (``tint``), plus how striped it is and how much of it
    is black.  Until 23 Sep 2026 this compared the median of the inner disc,
    plus the white fraction.  On camera-realistic balls that mistook 11-14% of
    single-ball detections for another ball, nearly all for the cue ball: a
    solid's white number circle, when it faces the camera, fills the middle of
    the ball, so the median turns white.  One detection in twenty of a ball
    then fell outside its *own* track's colour gate, which spawned a second
    track, and the two traded the ball back and forth.  Measured against the
    synthetic ground truth, this metric confuses 0.8-1.5%, and 95% of a ball's
    detections sit within 12 of its average against 60 before.

    Computed for a whole track x detection grid in one pass: one pair at a
    time meant a Python-level double loop on every frame.
    """
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)

    def columns(sigs: Sequence["ColorSignature"]):
        lab = np.array([s.ball_colour for s in sigs], dtype=np.float64).reshape(len(sigs), 3)
        stripe = np.array([s.stripe for s in sigs], dtype=np.float64)
        dark = np.array([s.dark_fraction for s in sigs], dtype=np.float64)
        return lab, stripe, dark

    a_lab, a_stripe, a_dark = columns(a)
    b_lab, b_stripe, b_dark = columns(b)
    delta = (a_lab[:, None, :] - b_lab[None, :, :]) * _LAB_WEIGHTS
    colour = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    colour += np.abs(a_stripe[:, None] - b_stripe[None, :]) * _STRIPE_WEIGHT
    colour += np.abs(a_dark[:, None] - b_dark[None, :]) * _DARK_WEIGHT
    return colour


def _unit_disc_offsets(count: int = 64) -> np.ndarray:
    """Evenly spread points on the unit disc (a sunflower/Vogel spiral).

    Sampling a fixed number of points instead of every pixel makes the cost of
    a colour signature independent of how large the ball is on screen, which
    matters because this runs for every detection on every frame.  A hundred
    samples is far more than a median and a fraction need.
    """
    k = np.arange(count, dtype=np.float64) + 0.5
    radius = np.sqrt(k / count)
    theta = k * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


_DISC_OFFSETS = _unit_disc_offsets()


def _disc_pixels(
    lab_image: np.ndarray, centre: Tuple[float, float], radius_px: float,
    exclude: Optional[np.ndarray] = None,
    neighbours: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Lab samples spread over the disc, minus any where ``exclude`` is set and
    any nearer one of ``neighbours`` than this centre."""
    h, w = lab_image.shape[:2]
    pts = _DISC_OFFSETS * radius_px + np.asarray(centre, dtype=np.float64)
    if neighbours:
        own = np.linalg.norm(pts - np.asarray(centre, dtype=np.float64), axis=1)
        keep = np.ones(len(pts), dtype=bool)
        for other in neighbours:
            keep &= np.linalg.norm(pts - np.asarray(other, dtype=np.float64), axis=1) >= own
        pts = pts[keep]
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[inside], ys[inside]
    if exclude is not None and xs.size:
        keep = exclude[ys, xs] == 0
        xs, ys = xs[keep], ys[keep]
    return lab_image[ys, xs].astype(np.float64)


#: A pixel is part of a ball's own colour if its chroma is at least this
#: fraction of the ball's high-percentile chroma (and at least the floor
#: below).  Relative, because a dark blue ball under broadcast light has a
#: chroma of 27 where a synthetic yellow one has 87.
_TINT_CHROMA_RATIO = 0.5
_TINT_CHROMA_FLOOR = 12.0
#: ...and a ball has a colour at all only if at least this much of it is
#: coloured.  A stripe's band is about half of it; the 8 and the cue ball have
#: none.
_MIN_COLOURED_FRACTION = 0.25
#: A pixel this far from the ball's own colour (weighted Lab, as in
#: ``ColorSignature.distance``) is cap, number circle or highlight.
_OFF_TINT_DISTANCE = 30.0
#: Black, for ``dark_fraction``: darker than any lit ball colour (the 2-ball,
#: the darkest, has lightness 66 on the sample clips) and colourless.
_DARK_LIGHTNESS = 58.0
_DARK_CHROMA = 16.0

#: A clump of up to this many balls is split cleanly enough to learn each
#: ball's colour from: see ``Detection.colour_ok``.
_CLEAN_CLUSTER_SIZE = 3


def sample_signature(
    lab_image: np.ndarray,
    centre: Tuple[float, float],
    radius_px: float,
    cloth_mask: Optional[np.ndarray] = None,
    neighbours: Optional[Sequence[Tuple[float, float]]] = None,
) -> ColorSignature:
    """Summarise the colour of the disc at ``centre``.

    Two samplings.  The identity the tracker matches on uses only the inner 62%
    of the ball: the rim is contaminated by the cloth behind it and by the
    dark occlusion shadow every ball casts.  The ball's own colour and how
    striped it is are measured out to 0.9 of the radius instead, because that
    is where a stripe's caps are when its band faces the camera -- with
    cloth-coloured pixels dropped, when the cloth mask is given, so that grey
    cloth at the rim is not mistaken for a cap -- and, for a ball split out of
    a cluster, with the pixels nearer one of its ``neighbours`` dropped too,
    so that the ball touching it is not mistaken for a stripe's cap either.
    """
    r = max(2.0, radius_px * 0.62)
    pixels = _disc_pixels(lab_image, centre, r)
    if pixels.shape[0] == 0:
        return ColorSignature.empty()

    lab = np.median(pixels, axis=0)
    chroma = np.linalg.norm(pixels[:, 1:] - 128.0, axis=1)
    white = np.count_nonzero((pixels[:, 0] > 165.0) & (chroma < 26.0)) / float(
        pixels.shape[0]
    )
    signature = ColorSignature(
        lab=lab,
        white_fraction=float(white),
        chroma=float(np.median(chroma)),
        chroma_high=float(np.percentile(chroma, 85)),
    )

    outer = _disc_pixels(lab_image, centre, max(2.0, radius_px * 0.9), cloth_mask, neighbours)
    if outer.shape[0] >= 8:
        c = np.linalg.norm(outer[:, 1:] - 128.0, axis=1)
        signature.dark_fraction = float(
            np.count_nonzero((outer[:, 0] < _DARK_LIGHTNESS) & (c < _DARK_CHROMA))
        ) / float(outer.shape[0])
        floor = max(_TINT_CHROMA_FLOOR, _TINT_CHROMA_RATIO * float(np.percentile(c, 85)))
        coloured = c >= floor
        if np.count_nonzero(coloured) >= max(6, _MIN_COLOURED_FRACTION * outer.shape[0]):
            tint = np.median(outer[coloured], axis=0)
            off = np.linalg.norm((outer - tint) * _LAB_WEIGHTS, axis=1) > _OFF_TINT_DISTANCE
            signature.tint = tint
            signature.stripe = float(np.count_nonzero(off)) / float(outer.shape[0])
        else:
            # A colourless ball -- the cue ball, the 8.  A handful of stray
            # coloured pixels at its rim is not its colour, and taking their
            # median made the 8's "colour" jump by 80-130 from one frame to the
            # next.  The median of the whole disc stays black on the 8 even
            # with its number circle in view, and white on the cue ball.
            signature.tint = np.median(outer, axis=0)
    return signature


#: Directions sampled around a candidate's rim.  Twenty-four is enough for the
#: median to survive a ball that is half hidden behind another.
_RIM_ANGLES = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
_RIM_COS = np.cos(_RIM_ANGLES)
_RIM_SIN = np.sin(_RIM_ANGLES)

#: Where the surroundings are sampled, in ball radii: just outside the rim, but
#: well inside where a touching neighbour's own centre would sit.
_RIM_RADII = (1.22, 1.42)


def rim_contrast(
    lab_image: np.ndarray,
    centre: Tuple[float, float],
    radius_px: float,
    signature: ColorSignature,
) -> float:
    """How strongly this disc's colour differs from what surrounds it.

    A ball *ends* at its rim.  One radius out there is cloth, or another ball,
    but never more of the same ball, so the colour step across the rim is large
    in almost every direction.

    A disc drawn inside something larger fails exactly here, which is what a
    bridge hand on the bed is.  Its knuckles are ball-thick, roughly ball-sized
    and convincingly round, so every shape test the detector applies says
    "ball"; what gives them away is that one radius further out there is simply
    more hand.  The median over directions is used rather than the mean so that
    a ball touching a neighbour, or clipped by the edge of the bed, still scores
    high.
    """
    h, w = lab_image.shape[:2]
    cx, cy = centre
    # Per direction, the larger step over the sampled radii; -1 marks a
    # direction that fell outside the image at every radius, which is dropped
    # rather than counted as "no edge here".
    best = np.full(_RIM_ANGLES.shape, -1.0)
    for frac in _RIM_RADII:
        xs = np.rint(cx + frac * radius_px * _RIM_COS).astype(np.int32)
        ys = np.rint(cy + frac * radius_px * _RIM_SIN).astype(np.int32)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(inside):
            continue
        around = lab_image[ys[inside], xs[inside]].astype(np.float64)
        delta = (around - signature.lab.astype(np.float64)) * _LAB_WEIGHTS
        # Sampling two radii keeps a ball whose cast shadow hugs one side from
        # reading as no edge at all.
        best[inside] = np.maximum(best[inside], np.linalg.norm(delta, axis=1))

    seen = best[best >= 0.0]
    if seen.size == 0:
        return 0.0
    return float(np.median(seen))


# --------------------------------------------------------------------------
# Detections
# --------------------------------------------------------------------------


def _colour_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Di Zenzo multi-channel gradient: direction and magnitude of colour change.

    Two touching balls of different colours can have almost no *brightness* step
    between them -- a blue ball against a red one at the same lightness gives a
    grayscale edge close to zero -- while the colour step is obvious.  Since the
    cluster splitter relies on each ball's circular edge voting for its own
    centre, edges it cannot see are balls it cannot find, which is precisely the
    situation inside a rack.

    The structure tensor J = sum_c (grad c)(grad c)^T is summed over channels;
    its dominant eigenvector is the direction of greatest colour change and the
    corresponding eigenvalue its strength.  The direction is only defined up to
    sign, which costs nothing here: the radial-symmetry vote is cast both ways.
    """
    if image.ndim == 2:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy, cv2.magnitude(gx, gy)

    gxx = np.zeros(image.shape[:2], dtype=np.float32)
    gyy = np.zeros(image.shape[:2], dtype=np.float32)
    gxy = np.zeros(image.shape[:2], dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        cx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        cy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gxx += cx * cx
        gyy += cy * cy
        gxy += cx * cy

    diff = gxx - gyy
    root = np.sqrt(diff * diff + 4.0 * gxy * gxy)
    lambda_max = 0.5 * (gxx + gyy + root)
    magnitude = np.sqrt(np.maximum(lambda_max, 0.0))

    theta = 0.5 * np.arctan2(2.0 * gxy, diff)
    return (
        (magnitude * np.cos(theta)).astype(np.float32),
        (magnitude * np.sin(theta)).astype(np.float32),
        magnitude.astype(np.float32),
    )


@dataclass
class Detection:
    centre_image: Tuple[float, float]
    centre_table: Tuple[float, float]
    radius_px: float
    area_ratio: float
    circularity: float
    signature: ColorSignature
    from_cluster: bool = False
    #: Median colour step across the rim -- see ``rim_contrast``.
    rim_contrast: float = 0.0
    #: Whether this detection's colour can be learned from.  A ball on its
    #: own, or one of a pair or three split apart: each of those is sampled
    #: only on its own side of the line to its neighbour.  In a bigger clump
    #: the balls hide each other, so its colours are not trusted.
    colour_ok: bool = True
    #: Seen partly past the bed's far edge, where only a ball already being
    #: followed is looked for -- see ``DetectorConfig.search_raised_bed``.
    in_raised_band: bool = False
    #: What the ball model (``billiards.ballnet``) makes of it, when it ran:
    #: the chance it is a ball, that it is the cue ball, of each colour
    #: family, and that it is a stripe.
    ball_p: float = 1.0
    cue_p: Optional[float] = None
    family_p: Optional[np.ndarray] = None
    stripe_p: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "x_px": round(self.centre_image[0], 2),
            "y_px": round(self.centre_image[1], 2),
            "x_in": round(self.centre_table[0], 3),
            "y_in": round(self.centre_table[1], 3),
            "radius_px": round(self.radius_px, 2),
            "area_ratio": round(self.area_ratio, 3),
            "circularity": round(self.circularity, 3),
            "from_cluster": self.from_cluster,
            "rim_contrast": round(self.rim_contrast, 1),
            "colour": self.signature.to_dict(),
        }


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class BallDetector:
    def __init__(self, cfg: Config, table: TableModel, cloth: ClothModel) -> None:
        self.cfg = cfg
        self.table = table
        self.cloth = cloth
        self._bed_mask: Optional[np.ndarray] = None
        self._bed_shape: Optional[Tuple[int, int]] = None
        #: The current frame's cloth mask, so a ball's colour can be sampled
        #: without the cloth showing round its rim.
        self._cloth_mask: Optional[np.ndarray] = None
        self._bed_interior: Optional[np.ndarray] = None
        #: The bed without the raised band, when that is searched too.
        self._plain_bed: Optional[np.ndarray] = None
        #: The band round the table (``surround_mask``), and its frame size.
        self._surround: Optional[np.ndarray] = None
        self._surround_shape: Optional[Tuple[int, int]] = None
        self.last_debug: dict = {}
        #: The learned check on each proposal, if there is one.
        self.ballnet = ballnet.load() if cfg.detector.ball_model != "off" else None
        #: Centres the cluster splitter found and then judged not to be balls.
        self._split_rejected: List[Tuple[float, float]] = []

    # -- masks -------------------------------------------------------------

    def bed_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Where balls are allowed to be: the bed, minus the pockets.

        Punching out the pockets is not cosmetic.  A pocket is a dark, round,
        roughly ball-sized hole that sits permanently on the playing surface, so
        to a "not cloth" detector it looks exactly like a ball that never moves.
        Without this, a pool table reports four to six phantom balls on every
        frame, each with its own track.
        """
        if self._bed_mask is None or self._bed_shape != shape[:2]:
            margin = (
                self.cfg.table.bed_margin_ball_diameters * self.table.ball_diameter_in
            )
            # Not widened to where balls *appear* (``raised_margin_in``): on the
            # sample broadcasts the calibrated far edge is the top of the far
            # cushion's face, which is cloth too, so the widened band reached
            # into the rail and the hands of a player leaning over it -- 30
            # tracks for 7 balls on albin_fedor.  A ball against the far
            # cushion therefore stays hard to see; see UPGRADE_NOTES.
            plain = self.table.bed_mask(shape, margin_in=margin)
            mask = plain
            if self.cfg.detector.search_raised_bed:
                mask = self.table.bed_mask(shape, margin_in=margin, raised_margin_in=margin)

            radius_in = (
                self.cfg.detector.pocket_exclusion_ball_diameters
                * self.table.ball_diameter_in
            )
            if radius_in > 0:
                pockets = self.table.pockets_table()
                # Draw the pocket as the *projection* of a circle on the cloth,
                # not as a circle in the image.  A pocket is flat, so at a
                # grazing angle its image is a markedly squashed ellipse; a
                # round exclusion zone big enough to cover it would also swallow
                # every ball resting near that rail.
                angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
                for pocket in pockets:
                    ring = np.column_stack(
                        [
                            pocket[0] + radius_in * np.cos(angles),
                            pocket[1] + radius_in * np.sin(angles),
                        ]
                    )
                    poly = self.table.table_to_image(ring)
                    if not np.all(np.isfinite(poly)):
                        continue
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
            self._bed_mask = mask
            self._plain_bed = cv2.bitwise_and(plain, mask) if plain is not mask else None
            self._bed_shape = shape[:2]
            # The bed pulled in by a couple of pixels: a blob with any pixel
            # outside this reaches the edge.  See ``_is_made_of_balls``.
            self._bed_interior = cv2.erode(mask, np.ones((5, 5), np.uint8))
        return self._bed_mask

    def bed_cloth_coverage(self, cloth_mask: np.ndarray) -> float:
        """Fraction of the bed polygon that still looks like cloth.

        Near 1.0 while the calibrated table is on screen, and it collapses the
        moment the broadcast cuts to another angle or a replay.  That makes it a
        direct, physically meaningful "is the table still where we think it is?"
        test, computed from a mask the detector needs anyway.
        """
        bed = self.bed_mask(cloth_mask.shape)
        bed_area = int(np.count_nonzero(bed))
        if bed_area == 0:
            return 0.0
        overlap = int(np.count_nonzero(cv2.bitwise_and(cloth_mask, bed)))
        return overlap / float(bed_area)

    def surround_mask(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """The band round the table beyond its cushions: the wooden rails and
        the floor, which are not cloth.  None if most of it is out of the
        picture, where it can say nothing.  See ``surround_cloth_coverage``."""
        if self._surround_shape != shape[:2]:
            self._surround_shape = shape[:2]
            self._surround = None
            h, w = shape[:2]
            outer = self.table.table_to_image(self.table.bed_polygon_table(-_SURROUND_OUTER_IN))
            inner = self.table.table_to_image(self.table.bed_polygon_table(-_SURROUND_INNER_IN))
            if np.all(np.isfinite(outer)) and np.all(np.isfinite(inner)):
                lim = 4.0 * max(h, w)
                outer, inner = np.clip(outer, -lim, lim), np.clip(inner, -lim, lim)
                band = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(band, np.round(outer).astype(np.int32), 255)
                cv2.fillConvexPoly(band, np.round(inner).astype(np.int32), 0)
                full = abs(cv2.contourArea(outer.astype(np.float32))) - abs(cv2.contourArea(inner.astype(np.float32)))
                if full > 0 and cv2.countNonZero(band) >= 0.3 * full:
                    self._surround = band
        return self._surround

    def surround_cloth_coverage(self, cloth_mask: np.ndarray) -> Optional[float]:
        """Fraction of the band round the table that looks like cloth.

        Near 0 while the camera is where it was calibrated.  If it pushes in,
        the cloth spreads out over where the rails and floor were; if it cuts
        to another view, anything can be there.  Together with the bed's
        coverage, that says whether this view still fits the picture without
        fitting the table's outline again -- which on a grey table, whose
        cushion tops and near rail come and go from the cloth's outline, was
        wrong often enough to be taken for the camera moving.
        """
        band = self.surround_mask(cloth_mask.shape)
        if band is None:
            return None
        area = cv2.countNonZero(band)
        return cv2.countNonZero(cv2.bitwise_and(cloth_mask, band)) / float(area) if area else None

    def foreground_mask(self, frame: np.ndarray, hsv: Optional[np.ndarray] = None,
                        cloth_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Everything on the bed that is not cloth."""
        if cloth_mask is None:
            if hsv is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cloth_mask = self.cloth.mask(hsv)

        r_px = self.table.expected_ball_radius_px(
            tuple(self.table.corners_image.mean(axis=0))
        )
        # Fill specular highlights *inside* the cloth before inverting.  A light
        # rig reflecting off the cloth is not a ball; closing with a kernel well
        # under one ball radius removes those without touching real balls.
        k_fill = max(3, int(round(r_px * 0.55)) | 1)
        cloth_mask = cv2.morphologyEx(
            cloth_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fill, k_fill)),
        )

        # Not healed first.  Where a stripe's band meets its caps the blur can
        # mix a colour inside the cloth's window -- a blue 10 on blue-grey cloth
        # -- drawing a line of "cloth" across the ball that the opening below
        # widens into two half-balls, each too small to count.  Closing such
        # cracks before the opening fixed that on the synthetic clip, and also
        # closed the gaps between a gloved player's fingers: on albin_fedor the
        # hand then passed for a black ball next to the 8, and five false
        # collisions followed.  See UPGRADE_NOTES.md.
        fg = cv2.bitwise_and(cv2.bitwise_not(cloth_mask), self.bed_mask(frame.shape))

        k_open = max(3, int(round(r_px * self.cfg.detector.open_radius_ball_radii)) | 1)
        fg = cv2.morphologyEx(
            fg,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)),
        )
        return fg

    # -- main entry point --------------------------------------------------

    def detect(self, frame: np.ndarray, hsv: Optional[np.ndarray] = None,
               cloth_mask: Optional[np.ndarray] = None) -> List[Detection]:
        if hsv is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if cloth_mask is None:
            cloth_mask = self.cloth.mask(hsv)
        fg = self.foreground_mask(frame, hsv, cloth_mask)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        self._cloth_mask = cloth_mask

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
        detections: List[Detection] = []
        rejected = {"area": 0, "shape": 0, "outside": 0, "rim": 0, "not_balls": 0}
        #: Turned down by the tests below; the ball model may overrule them.
        second_look: List[tuple] = []

        for i in range(1, num):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            r_expected = self.table.expected_ball_radius_px((float(cx), float(cy)))
            if r_expected <= 0.8:
                rejected["outside"] += 1
                continue
            expected_area = np.pi * r_expected * r_expected
            ratio = float(area) / float(expected_area)

            if ratio < self.cfg.detector.min_area_ratio or ratio > self.cfg.detector.max_area_ratio:
                rejected["area"] += 1
                continue

            component = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255

            if ratio >= self.cfg.detector.split_area_ratio:
                centres = self._split_cluster(
                    component, r_expected, offset=(x, y), colour=lab, area=int(area),
                    in_open=not self._touches_bed_edge(component, (x, y)),
                )
                if not centres:
                    rejected["not_balls"] += 1
                    kept = self._split_rejected
                    self._split_rejected = []
                    for c in kept:
                        second_look.append((c, r_expected, ratio, 1.0, True,
                                            [o for o in kept if o is not c], len(kept) <= _CLEAN_CLUSTER_SIZE))
                for c in centres:
                    others = [o for o in centres if o is not c]
                    det = self._make_detection(
                        c, r_expected, ratio, 1.0, lab, from_cluster=True,
                        neighbours=others,
                        colour_ok=len(centres) <= _CLEAN_CLUSTER_SIZE,
                    )
                    if det is None:
                        rejected["rim"] += 1
                        second_look.append((c, r_expected, ratio, 1.0, True, others,
                                            len(centres) <= _CLEAN_CLUSTER_SIZE))
                    else:
                        detections.append(det)
                continue

            shape_ok, circularity = self._check_shape(component)
            if not shape_ok:
                rejected["shape"] += 1
                if ratio <= self.cfg.detector.split_area_ratio:
                    second_look.append((self._refine_centre(component, offset=(x, y)), r_expected,
                                        ratio, circularity, False, None, True))
                continue

            centre = self._refine_centre(component, offset=(x, y))
            table_xy = self._table_point(centre)
            if table_xy is None:
                rejected["outside"] += 1
                continue
            det = self._make_detection(
                centre, r_expected, ratio, circularity, lab, from_cluster=False
            )
            if det is None:
                rejected["rim"] += 1
                second_look.append((centre, r_expected, ratio, circularity, False, None, True))
                continue
            detections.append(det)

        if self.ballnet is not None:
            detections = self._ask_model(frame, lab, detections, second_look, rejected)

        if self._plain_bed is not None:
            for det in detections:
                det.in_raised_band = (
                    self._outside_plain_bed(det) > self.cfg.detector.raised_band_outside_fraction
                )
            # Two balls cannot overlap.  One in the band on top of one on the
            # bed is the bed ball's own top, run into the dark line under the
            # cushion's nose -- a black 8 against the far rail split in two.
            on_bed = [np.asarray(d.centre_image) for d in detections if not d.in_raised_band]
            detections = [
                d for d in detections
                if not d.in_raised_band or all(
                    float(np.linalg.norm(np.asarray(d.centre_image) - c)) >= 1.6 * d.radius_px
                    for c in on_bed
                )
            ]

        if len(detections) > self.cfg.detector.max_detections:
            detections.sort(key=lambda d: abs(np.log(max(d.area_ratio, 1e-6))))
            detections = detections[: self.cfg.detector.max_detections]

        self.last_debug = {
            "foreground_mask": fg,
            "components": num - 1,
            "rejected": rejected,
            "accepted": len(detections),
        }
        return detections

    # -- the ball model ----------------------------------------------------

    def _ask_model(
        self,
        frame: np.ndarray,
        lab: np.ndarray,
        detections: List[Detection],
        second_look: Sequence[tuple],
        rejected: dict,
    ) -> List[Detection]:
        """Keep what the ball model agrees is a ball, and what it is sure is one.

        Every proposal the tests above accepted is shown to the model, and
        dropped if it is unlikely to be a ball: on the answer keys of 28 Sep
        2026 the tests passed a chalk cube on the rail, knuckles and pocket
        shadows as balls.  The proposals the tests turned down are shown too,
        and kept only if the model is sure: the green 6 on the 2026 US Open
        cast a shadow that made its blob too big for one ball, and was lost
        for seconds at a time.
        """
        cfg = self.cfg.detector
        extra: List[Detection] = []
        for centre, r_expected, ratio, circ, from_cluster, others, colour_ok in second_look:
            det = self._make_detection(centre, r_expected, ratio, circ, lab, from_cluster,
                                       neighbours=others, colour_ok=colour_ok, check_rim=False)
            if det is not None and all(
                (det.centre_image[0] - d.centre_image[0]) ** 2 + (det.centre_image[1] - d.centre_image[1]) ** 2
                >= (1.2 * d.radius_px) ** 2 for d in detections
            ):
                extra.append(det)
        every = detections + extra
        if not every:
            return detections
        kind, family, stripe = self.ballnet.score(
            frame, [d.centre_image for d in every], [d.radius_px for d in every]
        )
        kept: List[Detection] = []
        dropped = rescued = 0
        for i, det in enumerate(every):
            det.ball_p = float(1.0 - kind[i, 0])
            det.cue_p = float(kind[i, 1])
            det.family_p = family[i]
            det.stripe_p = float(stripe[i])
            if i < len(detections):
                if det.ball_p >= cfg.ball_model_reject:
                    kept.append(det)
                else:
                    dropped += 1
            elif det.ball_p >= cfg.ball_model_rescue:
                kept.append(det)
                rescued += 1
        rejected["model"] = dropped
        rejected["rescued"] = rescued
        return kept

    # -- helpers -----------------------------------------------------------

    def _outside_plain_bed(self, det: "Detection") -> float:
        """How much of a ball's disc lies past the bed, in the raised band."""
        plain = self._plain_bed
        if plain is None:
            return 0.0
        h, w = plain.shape[:2]
        pts = det.radius_px * 0.8 * _unit_disc_offsets(16) + np.asarray(det.centre_image)
        xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        return float(np.mean(plain[ys, xs] == 0))

    def _table_point(self, centre: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Where this image point sits on the bed, or None if it is off it."""
        table_pt = self.table.ball_image_to_table([centre])[0]
        table_xy = (float(table_pt[0]), float(table_pt[1]))
        margin = -0.75 * self.table.ball_radius_in  # allow slight overhang
        if not self.table.contains(table_xy, margin):
            return None
        return table_xy

    def _make_detection(
        self,
        centre: Tuple[float, float],
        r_expected: float,
        ratio: float,
        circularity: float,
        lab: np.ndarray,
        from_cluster: bool,
        neighbours: Optional[Sequence[Tuple[float, float]]] = None,
        colour_ok: bool = True,
        check_rim: bool = True,
    ) -> Optional[Detection]:
        table_xy = self._table_point(centre)
        if table_xy is None:
            return None
        # Radius recomputed at the refined centre: matters on wide-angle views.
        r = self.table.expected_ball_radius_px_table(table_xy)
        sig = sample_signature(lab, centre, r, self._cloth_mask, neighbours)

        # ...and it has to *stop* being that colour one radius further out.
        # Everything above this line is happy with any ball-sized round thing;
        # this is the test a knuckle fails.
        rim = rim_contrast(lab, centre, r, sig)
        if check_rim and rim < self.cfg.detector.rim_contrast_min:
            return None

        return Detection(
            centre_image=(float(centre[0]), float(centre[1])),
            centre_table=table_xy,
            radius_px=float(r),
            area_ratio=float(ratio),
            circularity=float(circularity),
            signature=sig,
            from_cluster=from_cluster,
            rim_contrast=rim,
            colour_ok=colour_ok,
        )

    def _touches_bed_edge(self, component: np.ndarray, offset: Tuple[int, int]) -> bool:
        """Does this blob reach the edge of the region searched for balls?"""
        inner = self._bed_interior
        if inner is None:
            return True
        h, w = component.shape[:2]
        x, y = offset
        window = inner[y : y + h, x : x + w]
        if window.shape != component.shape:
            return True
        return bool(np.any((component > 0) & (window == 0)))

    def _check_shape(self, component: np.ndarray) -> Tuple[bool, float]:
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False, 0.0
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if area <= 0 or perim <= 0:
            return False, 0.0

        circularity = float(4.0 * np.pi * area / (perim * perim))
        if circularity < self.cfg.detector.min_circularity:
            return False, circularity

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or area / hull_area < self.cfg.detector.min_solidity:
            return False, circularity

        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) <= 1e-6:
            return False, circularity
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > self.cfg.detector.max_aspect_ratio:
            return False, circularity
        return True, circularity

    def _refine_centre(
        self, component: np.ndarray, offset: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Sub-pixel centre from image moments, falling back to the bounding
        circle.  Moments beat ``minEnclosingCircle`` here because the enclosing
        circle is pinned by the two most extreme pixels and so jitters with
        every speckle on the silhouette."""
        m = cv2.moments(component, binaryImage=True)
        if m["m00"] > 0:
            return (
                offset[0] + m["m10"] / m["m00"],
                offset[1] + m["m01"] / m["m00"],
            )
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        (cx, cy), _ = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        return (offset[0] + cx, offset[1] + cy)

    # -- cluster splitting -------------------------------------------------

    def _split_cluster(
        self,
        component: np.ndarray,
        r_expected: float,
        offset: Tuple[int, int],
        colour: np.ndarray,
        area: int,
        in_open: bool = False,
    ) -> List[Tuple[float, float]]:
        """Separate touching balls inside one blob.

        Two complementary detectors are run and their results merged, because
        neither alone covers the cases that matter:

        * **Distance transform peaks** find balls that still have a sliver of
          cloth between them -- two balls just after contact, a loose cluster.
        * **Radial symmetry voting** finds balls with no gap at all.  A racked
          triangle is a single solid mass whose distance transform peaks in the
          middle of the *triangle*, nowhere near any ball, so the whole rack is
          invisible to the DT method.  Every ball still has a circular edge,
          though, and those edges vote for their own centre.

        Both are parameter-free in the way that matters: the ball radius comes
        from the table homography rather than from a number someone tuned.
        """
        pad = int(np.ceil(r_expected)) + 2
        padded = cv2.copyMakeBorder(
            component, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)

        # Gate the whole blob first: if nowhere inside it is more than about
        # half a ball radius from the outside, the blob is thinner than a ball
        # and simply cannot contain one.  This is what keeps the cue stick, the
        # bridge hand and rail glare out of the cluster splitter -- a purely
        # physical test, with no colour or length threshold to tune.
        if float(dist.max()) < self.cfg.detector.split_peak_min_ratio * r_expected:
            return []

        dt_centres, dt_scores = self._dt_peaks(dist, pad, r_expected, offset)
        rs_centres, rs_scores = self._radial_symmetry_peaks(
            component, r_expected, offset, colour
        )

        # Every radial-symmetry candidate must also sit somewhere thick enough
        # to hold a ball; voting alone can fire on a cap or a bright streak.
        keep_rs = []
        keep_rs_scores = []
        for centre, score in zip(rs_centres, rs_scores):
            yi = int(np.clip(round(centre[1] - offset[1] + pad), 0, dist.shape[0] - 1))
            xi = int(np.clip(round(centre[0] - offset[0] + pad), 0, dist.shape[1] - 1))
            if dist[yi, xi] >= 0.45 * r_expected:
                keep_rs.append(centre)
                keep_rs_scores.append(score)
        rs_centres, rs_scores = keep_rs, keep_rs_scores

        if not dt_centres and not rs_centres:
            return []

        # Normalise the two score scales before merging so neither dominates.
        merged: List[Tuple[Tuple[float, float], float]] = []
        for group in (list(zip(dt_centres, dt_scores)), list(zip(rs_centres, rs_scores))):
            if not group:
                continue
            top = max(s for _, s in group) or 1.0
            merged.extend((c, s / top) for c, s in group)

        merged.sort(key=lambda item: -item[1])
        min_sep = 1.45 * r_expected
        kept: List[Tuple[float, float]] = []
        for centre, _ in merged:
            if all(
                (centre[0] - k[0]) ** 2 + (centre[1] - k[1]) ** 2 >= min_sep * min_sep
                for k in kept
            ):
                kept.append(centre)

        # Never report more balls than could physically fit in the blob.
        max_balls = max(1, int(np.ceil(area / (0.68 * np.pi * r_expected**2))))
        kept = kept[:max_balls]

        if not self._is_made_of_balls(kept, dist, pad, r_expected, offset, colour, in_open):
            self._split_rejected = kept
            return []
        return kept

    def _is_made_of_balls(
        self,
        centres: Sequence[Tuple[float, float]],
        dist: np.ndarray,
        pad: int,
        r_expected: float,
        offset: Tuple[int, int],
        colour: np.ndarray,
        in_open: bool = False,
    ) -> bool:
        """Is this blob a group of balls, or a hand?

        Every individual check upstream passes on a bridge hand: its knuckles
        are ball-thick, so the distance-transform gate lets the blob in, and
        the splitter finds three convincing round peaks inside it.  Two things
        separate the cases, and a blob only has to manage one of them.

        **It accounts for itself.**  A group of touching balls is a union of
        discs of one known radius, so once the discs are drawn there should be
        nothing ball-thick left over.  What *is* left over may be thin -- a cue
        shaft, the cast shadow welding two balls together, a sleeve -- because
        none of those could hide a ball.  Three discs explain essentially all
        of a two-ball clump and about a third of a hand.

        **Or its discs sit on real ball edges.**  Failing the first test does
        not prove the blob is not balls: it may be balls the splitter could not
        separate.  A racked triangle of same-coloured neighbours gives five of
        eight, so its discs cover 0.60 of it -- and rejecting on that alone
        cost nine points of recall against ground truth.  But those five sit on
        unmistakable circular edges, which the knuckles do not.  The blob is
        judged as a whole, by the median, because a blob is a clump of balls or
        it is not; one ball and two knuckles is not a thing.
        """
        cfg = self.cfg.detector
        if cfg.cluster_core_coverage_min <= 0.0:
            return True
        if not centres:
            return False

        # The blob's ball-thick core: anywhere a ball could actually be hiding.
        core = dist >= 0.5 * r_expected
        core_area = int(np.count_nonzero(core))
        if core_area == 0:
            return False

        explained = np.zeros(dist.shape, dtype=np.uint8)
        radius = max(1, int(round(r_expected)))
        for cx, cy in centres:
            cv2.circle(
                explained,
                (int(round(cx - offset[0] + pad)), int(round(cy - offset[1] + pad))),
                radius,
                255,
                -1,
            )
        covered = int(np.count_nonzero(core & (explained > 0)))
        if covered / float(core_area) >= cfg.cluster_core_coverage_min:
            return True

        rims = [
            rim_contrast(colour, c, r_expected, sample_signature(colour, c, r_expected))
            for c in centres
        ]
        # A blob out in the open, clear of the bed's edge, cannot be a hand: a
        # hand, a forearm or a cue reaches the table from outside it, so its
        # blob crosses the edge.  On the three sample clips every rejected blob
        # of two or more candidates touched the edge but two, with rims of 6
        # and 13, while a static rack of camera-realistic balls -- same-looking
        # neighbours, no gap between them -- sits in the open with rims of
        # 33-51.  Such a rack failed both tests above and went unseen until
        # the break; the looser bar for a blob in the open finds it.
        if in_open and len(centres) >= 2:
            return float(np.median(rims)) >= cfg.cluster_open_rim_contrast_min
        return float(np.median(rims)) >= cfg.cluster_rim_contrast_min

    def _dt_peaks(
        self,
        dist: np.ndarray,
        pad: int,
        r_expected: float,
        offset: Tuple[int, int],
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Local maxima of the distance transform, height-gated to ball size."""
        if dist.max() <= 0:
            return [], []

        sep = max(
            3, int(round(r_expected * self.cfg.detector.peak_separation_ball_radii)) | 1
        )
        dilated = cv2.dilate(
            dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep, sep))
        )
        # The distance transform of a disc peaks at exactly its radius, so a
        # peak far outside that band belongs to something wider than a ball --
        # a forearm, a sleeve, a bundle of cues on the bed.
        lo = self.cfg.detector.split_peak_min_ratio * r_expected
        hi = self.cfg.detector.split_peak_max_ratio * r_expected
        peak_mask = ((dist >= dilated - 1e-4) & (dist >= lo) & (dist <= hi)).astype(
            np.uint8
        )
        if not np.any(peak_mask):
            return [], []

        n, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask, 8)
        centres: List[Tuple[float, float]] = []
        scores: List[float] = []
        for i in range(1, n):
            cx, cy = centroids[i]
            yi = int(np.clip(round(cy), 0, dist.shape[0] - 1))
            xi = int(np.clip(round(cx), 0, dist.shape[1] - 1))
            centres.append((cx - pad + offset[0], cy - pad + offset[1]))
            scores.append(float(dist[yi, xi]))
        return centres, scores

    def _radial_symmetry_peaks(
        self,
        component: np.ndarray,
        r_expected: float,
        offset: Tuple[int, int],
        colour: np.ndarray,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Fast radial symmetry voting at the known ball radius.

        Every edge pixel casts a vote one ball-radius along its gradient, in
        both directions.  A circular edge of the right size therefore piles all
        of its votes onto its own centre, while straight edges (a cue, a rail)
        smear their votes along a line and never accumulate.  Because the radius
        is supplied by the homography, this has no free scale parameter -- it is
        looking for objects of one specific physical size.
        """
        h, w = component.shape[:2]
        pad = int(np.ceil(r_expected)) + 3
        x0 = max(0, offset[0] - pad)
        y0 = max(0, offset[1] - pad)
        x1 = min(colour.shape[1], offset[0] + w + pad)
        y1 = min(colour.shape[0], offset[1] + h + pad)
        roi = colour[y0:y1, x0:x1]
        if roi.size == 0 or min(roi.shape[:2]) < 5:
            return [], []

        roi_blur = cv2.GaussianBlur(roi, (0, 0), max(0.8, 0.14 * r_expected))
        gx, gy, mag = _colour_gradient(roi_blur)
        peak_mag = float(mag.max())
        if peak_mag <= 1e-6:
            return [], []

        # Adaptive edge threshold: a fixed Canny-style number would be exactly
        # the kind of per-video knob this rewrite exists to remove.
        thresh = max(float(np.percentile(mag, 75)), 0.12 * peak_mag)
        ys, xs = np.nonzero(mag >= thresh)
        if xs.size < 24:
            return [], []

        m = mag[ys, xs]
        ux = gx[ys, xs] / m
        uy = gy[ys, xs] / m

        rh, rw = roi.shape[:2]
        votes = np.zeros((rh, rw), dtype=np.float32)
        for sign in (1.0, -1.0):
            vx = np.round(xs + sign * r_expected * ux).astype(np.int32)
            vy = np.round(ys + sign * r_expected * uy).astype(np.int32)
            inside = (vx >= 0) & (vx < rw) & (vy >= 0) & (vy < rh)
            if not np.any(inside):
                continue
            np.add.at(votes, (vy[inside], vx[inside]), m[inside])

        votes = cv2.GaussianBlur(votes, (0, 0), max(1.0, 0.28 * r_expected))

        # A ball centre has to be inside the blob it came from.
        mask_full = np.zeros((rh, rw), dtype=np.uint8)
        oy, ox = offset[1] - y0, offset[0] - x0
        mask_full[oy : oy + h, ox : ox + w] = component
        votes[mask_full == 0] = 0.0

        best = float(votes.max())
        if best <= 1e-6:
            return [], []

        sep = max(3, int(round(1.5 * r_expected)) | 1)
        dilated = cv2.dilate(
            votes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep, sep))
        )
        peak_mask = ((votes >= dilated - 1e-6) & (votes >= 0.30 * best)).astype(np.uint8)
        if not np.any(peak_mask):
            return [], []

        n, _, _, centroids = cv2.connectedComponentsWithStats(peak_mask, 8)
        centres: List[Tuple[float, float]] = []
        scores: List[float] = []
        for i in range(1, n):
            cx, cy = centroids[i]
            yi = int(np.clip(round(cy), 0, rh - 1))
            xi = int(np.clip(round(cx), 0, rw - 1))
            centres.append((cx + x0, cy + y0))
            scores.append(float(votes[yi, xi]))
        return centres, scores
