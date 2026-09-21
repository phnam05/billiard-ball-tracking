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

    @staticmethod
    def empty() -> "ColorSignature":
        return ColorSignature(lab=np.array([128.0, 128.0, 128.0]), white_fraction=0.0)

    def distance(self, other: "ColorSignature") -> float:
        d_lab = self.lab.astype(np.float64) - other.lab.astype(np.float64)
        colour = float(np.linalg.norm(_LAB_WEIGHTS * d_lab))
        stripe = abs(self.white_fraction - other.white_fraction) * _STRIPE_WEIGHT
        return colour + stripe

    def blend(self, other: "ColorSignature", alpha: float) -> "ColorSignature":
        """Exponential moving average towards ``other``."""
        return ColorSignature(
            lab=(1.0 - alpha) * self.lab + alpha * other.lab,
            white_fraction=(1.0 - alpha) * self.white_fraction
            + alpha * other.white_fraction,
            chroma=(1.0 - alpha) * self.chroma + alpha * other.chroma,
            chroma_high=(1.0 - alpha) * self.chroma_high + alpha * other.chroma_high,
        )

    @property
    def bgr(self) -> Tuple[int, int, int]:
        """Approximate display colour for this signature."""
        patch = np.zeros((1, 1, 3), dtype=np.uint8)
        patch[0, 0] = np.clip(self.lab, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(patch, cv2.COLOR_Lab2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def classify(self) -> str:
        """Coarse ball type: ``cue``, ``stripe``, ``eight`` or ``solid``.

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
            "white_fraction": round(float(self.white_fraction), 3),
            "chroma": round(float(self.chroma), 1),
            "chroma_high": round(float(self.chroma_high), 1),
            "type": self.classify(),
        }


#: Per-channel weights used by ColorSignature.distance.  Lightness counts for
#: less because it is the channel that moves when a ball rolls through a shadow.
_LAB_WEIGHTS = np.array([0.45, 1.0, 1.0])
_STRIPE_WEIGHT = 26.0


def colour_distance_matrix(
    a: Sequence["ColorSignature"], b: Sequence["ColorSignature"]
) -> np.ndarray:
    """All pairwise colour distances at once.

    Same metric as ``ColorSignature.distance``, but computed for a whole
    track x detection grid in one pass.  Doing it one pair at a time meant a
    Python-level double loop plus several small array allocations on every
    frame, which was one of the larger costs in the tracker.
    """
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)

    a_lab = np.array([s.lab for s in a], dtype=np.float64).reshape(len(a), 3)
    b_lab = np.array([s.lab for s in b], dtype=np.float64).reshape(len(b), 3)
    a_white = np.array([s.white_fraction for s in a], dtype=np.float64)
    b_white = np.array([s.white_fraction for s in b], dtype=np.float64)

    delta = (a_lab[:, None, :] - b_lab[None, :, :]) * _LAB_WEIGHTS
    colour = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    stripe = np.abs(a_white[:, None] - b_white[None, :]) * _STRIPE_WEIGHT
    return colour + stripe


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


def sample_signature(
    lab_image: np.ndarray, centre: Tuple[float, float], radius_px: float
) -> ColorSignature:
    """Summarise the colour of the disc at ``centre``.

    Only the inner 62% of the ball is sampled: the rim is contaminated by the
    cloth behind it and by the dark occlusion shadow every ball casts.
    """
    h, w = lab_image.shape[:2]
    r = max(2.0, radius_px * 0.62)

    pts = _DISC_OFFSETS * r + np.asarray(centre, dtype=np.float64)
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(inside):
        return ColorSignature.empty()

    pixels = lab_image[ys[inside], xs[inside]].astype(np.float64)
    lab = np.median(pixels, axis=0)
    chroma = np.linalg.norm(pixels[:, 1:] - 128.0, axis=1)
    white = np.count_nonzero((pixels[:, 0] > 165.0) & (chroma < 26.0)) / float(
        pixels.shape[0]
    )
    return ColorSignature(
        lab=lab,
        white_fraction=float(white),
        chroma=float(np.median(chroma)),
        chroma_high=float(np.percentile(chroma, 85)),
    )


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
        self.last_debug: dict = {}

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
            mask = self.table.bed_mask(shape, margin_in=margin)

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
            self._bed_shape = shape[:2]
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
        fg = self.foreground_mask(frame, hsv, cloth_mask)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
        detections: List[Detection] = []
        rejected = {"area": 0, "shape": 0, "outside": 0}

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
                    component, r_expected, offset=(x, y), colour=lab, area=int(area)
                )
                for c in centres:
                    det = self._make_detection(
                        c, r_expected, ratio, 1.0, lab, from_cluster=True
                    )
                    if det is not None:
                        detections.append(det)
                continue

            shape_ok, circularity = self._check_shape(component)
            if not shape_ok:
                rejected["shape"] += 1
                continue

            centre = self._refine_centre(component, offset=(x, y))
            det = self._make_detection(
                centre, r_expected, ratio, circularity, lab, from_cluster=False
            )
            if det is None:
                rejected["outside"] += 1
                continue
            detections.append(det)

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

    # -- helpers -----------------------------------------------------------

    def _make_detection(
        self,
        centre: Tuple[float, float],
        r_expected: float,
        ratio: float,
        circularity: float,
        lab: np.ndarray,
        from_cluster: bool,
    ) -> Optional[Detection]:
        table_pt = self.table.image_to_table([centre])[0]
        table_xy = (float(table_pt[0]), float(table_pt[1]))
        margin = -0.75 * self.table.ball_radius_in  # allow slight overhang
        if not self.table.contains(table_xy, margin):
            return None
        # Radius recomputed at the refined centre: matters on wide-angle views.
        r = self.table.expected_ball_radius_px_table(table_xy)
        sig = sample_signature(lab, centre, r)
        return Detection(
            centre_image=(float(centre[0]), float(centre[1])),
            centre_table=table_xy,
            radius_px=float(r),
            area_ratio=float(ratio),
            circularity=float(circularity),
            signature=sig,
            from_cluster=from_cluster,
        )

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
        return kept[:max_balls]

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
