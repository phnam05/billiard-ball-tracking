"""Table geometry: quadrilateral fitting, homography, and physical scale.

The single most useful idea in the rewrite lives here: once we know the
homography from the camera image to the table's own coordinate system (inches),
every threshold in the rest of the pipeline can be stated in inches instead of
pixels.  That is what removes the endless per-video retuning.

Table coordinates are ``(x, y)`` in inches with the origin at one corner of the
playing surface, ``x`` along the long axis (0..length_in) and ``y`` along the
short axis (0..width_in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Corner extraction
# --------------------------------------------------------------------------


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Ordering is done by angle around the centroid, which is correct for any
    convex quadrilateral including the strong trapezoids a low camera produces.
    The naive "smallest x+y is top-left" rule fails on a rotated table; sorting
    by angle does not.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("order_corners expects exactly 4 points")

    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(angles)
    pts = pts[order]  # counter-clockwise in maths convention = clockwise on screen

    # Rotate so the first point is the one closest to the image origin.
    start = int(np.argmin(np.linalg.norm(pts - pts.min(axis=0), axis=1)))
    pts = np.roll(pts, -start, axis=0)

    # Screen y grows downwards, so sorting by atan2 gives clockwise order:
    # TL, TR, BR, BL.  Verify and flip if the winding came out the other way.
    if _signed_area(pts) < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts


def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    """Scalar cross product of two 2-D vectors.

    NumPy 2.0 removed ``np.cross`` for 2-D inputs, so this is spelled out.
    """
    return float(u[0] * v[1] - u[1] * v[0])


def _line_from_points(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points."""
    return np.cross(np.array([p[0], p[1], 1.0]), np.array([q[0], q[1], 1.0]))


def _intersect(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None  # parallel
    return np.array([p[0] / p[2], p[1] / p[2]])


def _refine_edge_line(
    points: np.ndarray, p: np.ndarray, q: np.ndarray, tol: float
) -> np.ndarray:
    """Re-fit a cushion's supporting line to the contour points along it.

    Taking the line through the two endpoints of the simplified polygon is not
    good enough: at a pool table both ends of every cushion are bitten off by a
    pocket, and the two bites are different sizes under perspective, which tilts
    the line and pushes the reconstructed corner inwards by a couple of inches.

    The cushion's *middle* is unspoiled, and it is most of the edge, so a robust
    fit over all contour points near the edge recovers the true line and with it
    the true corner.
    """
    direction = q - p
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return _line_from_points(p, q)
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]])

    rel = points - p
    along = rel @ direction
    perp = rel @ normal
    # Trim the ends of the span: the simplified polygon's endpoints sit right
    # where the pocket starts eating the cushion, so the last few percent at
    # each end is exactly the contaminated part.
    inliers = points[
        (np.abs(perp) < tol) & (along > 0.04 * norm) & (along < 0.96 * norm)
    ]
    if inliers.shape[0] < 12:
        return _line_from_points(p, q)

    # Two rounds of robust refitting: fit, drop the points the pockets bent, refit.
    for _ in range(2):
        vx, vy, x0, y0 = cv2.fitLine(
            inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        d = np.abs((inliers[:, 0] - x0) * vy - (inliers[:, 1] - y0) * vx)
        keep = d <= max(1.0, 2.5 * float(np.median(d)) + 1.0)
        if np.count_nonzero(keep) < 12:
            break
        inliers = inliers[keep]

    vx, vy, x0, y0 = cv2.fitLine(
        inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    a = np.array([x0, y0], dtype=np.float64)
    b = a + np.array([vx, vy], dtype=np.float64)
    return _line_from_points(a, b)


def quad_from_longest_edges(contour: np.ndarray) -> Optional[np.ndarray]:
    """Fit a quadrilateral by intersecting the supporting lines of the four
    dominant edges of the contour.

    This is the right primitive for a pool table: the four cushions are long
    straight edges, while pocket jaws, the referee's hand and cloth wrinkles
    create many short ones.  Selecting by edge *length* and then intersecting
    the infinite lines recovers the true corners even when the actual corner is
    occluded or rounded off by a pocket -- which the old
    "contour point nearest the image corner" rule could never do.
    """
    hull = cv2.convexHull(contour.astype(np.int32))
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    approx = cv2.approxPolyDP(hull, 0.008 * peri, True).reshape(-1, 2).astype(np.float64)
    if approx.shape[0] < 4:
        return None

    n = approx.shape[0]
    edges = []
    for i in range(n):
        p, q = approx[i], approx[(i + 1) % n]
        length = float(np.linalg.norm(q - p))
        angle = float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180.0)
        edges.append({"i": i, "p": p, "q": q, "len": length, "angle": angle})

    edges.sort(key=lambda e: e["len"], reverse=True)
    base_angle = edges[0]["angle"]

    def angle_delta(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    group_a = [e for e in edges if angle_delta(e["angle"], base_angle) < 40.0]
    group_b = [e for e in edges if angle_delta(e["angle"], base_angle) >= 40.0]
    if len(group_a) < 2 or len(group_b) < 2:
        return None

    chosen = group_a[:2] + group_b[:2]
    chosen.sort(key=lambda e: e["i"])  # restore travel order around the polygon

    raw_points = contour.reshape(-1, 2).astype(np.float64)
    tol = max(4.0, 0.02 * peri / 4.0)
    lines = [_refine_edge_line(raw_points, e["p"], e["q"], tol) for e in chosen]
    corners: List[np.ndarray] = []
    for i in range(4):
        pt = _intersect(lines[i], lines[(i + 1) % 4])
        if pt is None:
            return None
        corners.append(pt)
    quad = np.array(corners, dtype=np.float64)

    if not _is_sane_quad(quad, cv2.contourArea(hull)):
        return None
    return order_corners(quad)


def _is_sane_quad(quad: np.ndarray, reference_area: float) -> bool:
    """Reject degenerate fits: self-intersecting, tiny, or wildly oversized."""
    if not np.all(np.isfinite(quad)):
        return False
    area = abs(_signed_area(quad))
    if area <= 0 or reference_area <= 0:
        return False
    if not (0.55 * reference_area <= area <= 2.2 * reference_area):
        return False
    # Convexity check: all cross products must share a sign.
    ordered = order_corners(quad)
    signs = []
    for i in range(4):
        a, b, c = ordered[i], ordered[(i + 1) % 4], ordered[(i + 2) % 4]
        signs.append(np.sign(_cross2(b - a, c - b)))
    return len(set(s for s in signs if s != 0)) == 1


def quad_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort quadrilateral for a table contour, with two fallbacks."""
    quad = quad_from_longest_edges(contour)
    if quad is not None:
        return quad

    # Fallback 1: adaptive Douglas-Peucker until exactly 4 vertices appear.
    hull = cv2.convexHull(contour.astype(np.int32))
    peri = cv2.arcLength(hull, True)
    lo, hi = 0.001, 0.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull, mid * peri, True)
        if len(approx) > 4:
            lo = mid
        elif len(approx) < 4:
            hi = mid
        else:
            return order_corners(approx.reshape(-1, 2).astype(np.float64))

    # Fallback 2: minimum-area rotated rectangle.  Wrong under strong
    # perspective, but far better than giving up.
    rect = cv2.minAreaRect(contour.astype(np.int32))
    box = cv2.boxPoints(rect).astype(np.float64)
    return order_corners(box)


# --------------------------------------------------------------------------
# The table model
# --------------------------------------------------------------------------


def focal_lengths_for_homography(
    corners_image: np.ndarray, dst_table: np.ndarray, principal: Tuple[float, float]
) -> Tuple[float, float]:
    """Two independent estimates of the squared focal length, in px^2.

    For a homography ``H = K [r1 r2 t]`` mapping a world plane to the image, the
    rotation columns are orthonormal.  That gives two constraints,
    ``r1 . r2 = 0`` and ``|r1| = |r2|``, each of which pins down ``f`` on its
    own.  If the plane really is a rectangle of the assumed shape, both come out
    positive and agree with each other.  If the assumed shape is wrong -- for
    instance a 2:1 table matched to its own corners the other way round -- they
    do not.

    Returns ``(f_squared_from_orthogonality, f_squared_from_equal_norms)``, with
    NaN where the expression is degenerate (a fronto-parallel view, where the
    focal length genuinely cannot be recovered).

    Only the **second** value can tell one table orientation from the other.
    The orthogonality expression is invariant under swapping and rescaling the
    two columns, which is exactly what changing the assumed orientation does, so
    it returns the same number either way.  The equal-norms constraint is the
    one that depends on the assumed *aspect ratio*, and therefore the one that
    knows whether the long side of the table runs across the frame or into it.
    """
    H = cv2.getPerspectiveTransform(
        dst_table.astype(np.float32), corners_image.astype(np.float32)
    ).astype(np.float64)
    shift = np.array(
        [[1.0, 0.0, -principal[0]], [0.0, 1.0, -principal[1]], [0.0, 0.0, 1.0]]
    )
    Hn = shift @ H
    h1, h2 = Hn[:, 0], Hn[:, 1]

    den_a = h1[2] * h2[2]
    f2_a = (
        -(h1[0] * h2[0] + h1[1] * h2[1]) / den_a if abs(den_a) > 1e-12 else np.nan
    )

    den_b = h1[2] ** 2 - h2[2] ** 2
    f2_b = (
        ((h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)) / den_b
        if abs(den_b) > 1e-12
        else np.nan
    )
    return float(f2_a), float(f2_b)


@dataclass
class TableModel:
    """A calibrated table: image corners plus the image<->table homography."""

    corners_image: np.ndarray  # (4,2) float, TL TR BR BL in image pixels
    length_in: float
    width_in: float
    ball_diameter_in: float
    has_pockets: bool = True
    #: (width, height) of the frame these corners came from.  Used to place the
    #: principal point when deciding which way round the table is.
    image_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        self.corners_image = np.asarray(self.corners_image, dtype=np.float64).reshape(4, 2)
        dst = self._table_corner_targets()
        self.H = cv2.getPerspectiveTransform(
            self.corners_image.astype(np.float32), dst.astype(np.float32)
        )
        self.H_inv = np.linalg.inv(self.H)

    # -- orientation -------------------------------------------------------

    def _orientation_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        L, W = self.length_in, self.width_in
        along_top = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
        along_side = np.array([[0, 0], [0, W], [L, W], [L, 0]], dtype=np.float64)
        return along_top, along_side

    def _table_corner_targets(self) -> np.ndarray:
        """Where the four image corners land in table inches.

        Deciding which image axis is the table's *long* axis is not the trivial
        question it looks like.  The obvious rule -- "the physically longer side
        subtends more pixels" -- is wrong exactly when it matters most: filmed
        from behind one end rail, which is the standard pool broadcast angle,
        the 100-inch length is foreshortened to fewer pixels than the 50-inch
        near cushion.  Following that rule there gets the scale wrong by a
        factor of two, which then makes every ball appear twice its expected
        size, so each one is treated as a cluster and split in half.

        So instead: try both assignments, and keep the one that a real camera
        could actually have produced.  A homography onto a rectangle of the
        right shape yields a positive, self-consistent focal length; the wrong
        shape does not.  The pixel-span rule survives only as a fallback for
        near-fronto-parallel views, where the focal length is genuinely not
        recoverable -- and where, conveniently, the rule is also correct.
        """
        along_top, along_side = self._orientation_candidates()

        if self.image_size is not None:
            principal = (self.image_size[0] / 2.0, self.image_size[1] / 2.0)
            diag = float(np.hypot(*self.image_size))
        else:
            principal = tuple(self.corners_image.mean(axis=0))
            span = self.corners_image.max(axis=0) - self.corners_image.min(axis=0)
            diag = float(np.hypot(*span))

        # Plausible focal lengths for real footage span roughly a fisheye to a
        # long broadcast lens.  Anything outside that is a numerical artefact.
        f_lo, f_hi = 0.25 * diag, 25.0 * diag
        reference = 1.1 * diag  # a fairly ordinary lens, used only to break ties

        viable = []
        for dst in (along_top, along_side):
            _, f2 = focal_lengths_for_homography(self.corners_image, dst, principal)
            if not np.isfinite(f2) or f2 <= 0:
                continue
            f = float(np.sqrt(f2))
            if not (f_lo <= f <= f_hi):
                continue
            viable.append((abs(np.log(f / reference)), dst))

        if viable:
            viable.sort(key=lambda item: item[0])
            return viable[0][1]

        # Fronto-parallel or numerically degenerate: the focal length is not
        # recoverable, but in that regime foreshortening is weak and the simple
        # pixel-span rule is reliable.
        tl, tr, br, bl = self.corners_image
        span_top_bottom = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
        span_left_right = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
        return along_top if span_top_bottom >= span_left_right else along_side

    # -- coordinate transforms --------------------------------------------

    @property
    def ball_radius_in(self) -> float:
        return self.ball_diameter_in / 2.0

    def image_to_table(self, pts: Sequence[Point] | np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def table_to_image(self, pts: Sequence[Point] | np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        out = cv2.perspectiveTransform(pts, self.H_inv)
        return out.reshape(-1, 2)

    # -- perspective-aware scale ------------------------------------------

    # -- cached scale field ------------------------------------------------
    #
    # The local magnification varies smoothly across the table, so evaluating
    # it exactly for every detection on every frame -- two perspective
    # transforms and an SVD apiece -- is wasted work.  It is tabulated once on a
    # grid in table coordinates and interpolated, which is both faster and
    # exactly as accurate at the scale anything here cares about.

    _GRID_NX = 65
    _GRID_NY = 33

    def _scale_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, "_scale_grid_cache", None)
        if cached is not None:
            return cached

        eps = 0.5
        xs = np.linspace(0.0, self.length_in, self._GRID_NX)
        ys = np.linspace(0.0, self.width_in, self._GRID_NY)
        gx, gy = np.meshgrid(xs, ys)
        base = np.column_stack([gx.ravel(), gy.ravel()])

        img = self.table_to_image(base)
        img_dx = self.table_to_image(base + np.array([eps, 0.0]))
        img_dy = self.table_to_image(base + np.array([0.0, eps]))

        a = (img_dx[:, 0] - img[:, 0]) / eps
        c = (img_dx[:, 1] - img[:, 1]) / eps
        b = (img_dy[:, 0] - img[:, 0]) / eps
        d = (img_dy[:, 1] - img[:, 1]) / eps

        # Closed-form singular values of the 2x2 Jacobian [[a, b], [c, d]].
        e = 0.5 * (a * a + b * b + c * c + d * d)
        f = 0.5 * (a * a + b * b - c * c - d * d)
        g = a * c + b * d
        root = np.sqrt(np.maximum(f * f + g * g, 0.0))
        sigma_max = np.sqrt(np.maximum(e + root, 1e-12))
        area = np.sqrt(np.abs(a * d - b * c))

        sphere = sigma_max.reshape(self._GRID_NY, self._GRID_NX)
        flat = area.reshape(self._GRID_NY, self._GRID_NX)
        self._scale_grid_cache = (sphere, flat)
        return self._scale_grid_cache

    def _sample_grid(self, grid: np.ndarray, table_pt: Point) -> float:
        """Bilinear lookup, clamped to the table domain."""
        u = float(table_pt[0]) / max(self.length_in, 1e-9) * (self._GRID_NX - 1)
        v = float(table_pt[1]) / max(self.width_in, 1e-9) * (self._GRID_NY - 1)
        u = min(max(u, 0.0), self._GRID_NX - 1.0)
        v = min(max(v, 0.0), self._GRID_NY - 1.0)

        i0 = int(u)
        j0 = int(v)
        i1 = min(i0 + 1, self._GRID_NX - 1)
        j1 = min(j0 + 1, self._GRID_NY - 1)
        fu = u - i0
        fv = v - j0

        top = grid[j0, i0] * (1.0 - fu) + grid[j0, i1] * fu
        bottom = grid[j1, i0] * (1.0 - fu) + grid[j1, i1] * fu
        return float(max(top * (1.0 - fv) + bottom * fv, 1e-6))

    def sphere_scale_at_table(self, table_pt: Point) -> float:
        return self._sample_grid(self._scale_grids()[0], table_pt)

    def px_per_inch_at_table(self, table_pt: Point) -> float:
        return self._sample_grid(self._scale_grids()[1], table_pt)

    def expected_ball_radius_px_table(self, table_pt: Point) -> float:
        return self.ball_radius_in * self.sphere_scale_at_table(table_pt)

    def local_jacobian(self, image_pt: Point, eps: float = 0.5) -> np.ndarray:
        """2x2 derivative of the table->image map at an image point.

        Its singular values are the local magnification along the two principal
        directions of the projection.  Under a grazing camera angle they differ
        by a large factor: one inch across the view covers far more pixels than
        one inch into it.
        """
        t = self.image_to_table([image_pt])[0]
        probe = np.array(
            [t, t + (eps, 0.0), t + (0.0, eps)], dtype=np.float64
        )
        back = self.table_to_image(probe)
        return np.column_stack(
            [(back[1] - back[0]) / eps, (back[2] - back[0]) / eps]
        )

    def px_per_inch_at(self, image_pt: Point) -> float:
        """Area-preserving local scale, in pixels per table inch.

        This is ``sqrt(|det J|)``, the right scale for a **flat** feature lying
        on the cloth -- a pocket, a spot, a marking.
        """
        t = self.image_to_table([image_pt])[0]
        return self.px_per_inch_at_table((float(t[0]), float(t[1])))

    def sphere_scale_at(self, image_pt: Point) -> float:
        """Local scale governing the apparent size of a **sphere**, in px/inch.

        A billiard ball is not a disc painted on the cloth, and the difference
        matters more than it sounds.  A flat disc viewed at a grazing angle is
        squashed into a thin ellipse; a sphere is not squashed at all -- its
        silhouette is always a circle, of angular radius R/d, so its image
        radius follows the magnification *across* the line of sight rather than
        the foreshortened one.

        Using the flat-disc scale therefore under-predicts ball size badly on
        exactly the camera angle pool is normally filmed from (behind one end
        rail, looking down the table).  Measured on real broadcast footage, it
        was out by 40-80%, which made every single ball look like a cluster
        roughly three times its expected area -- so the detector "split" each
        ball into two phantom halves.

        The largest singular value of the local Jacobian is that across-view
        magnification.
        """
        t = self.image_to_table([image_pt])[0]
        return self.sphere_scale_at_table((float(t[0]), float(t[1])))

    def expected_ball_radius_px(self, image_pt: Point) -> float:
        return self.ball_radius_in * self.sphere_scale_at(image_pt)

    def mean_px_per_inch(self) -> float:
        centre = self.corners_image.mean(axis=0)
        return self.px_per_inch_at((float(centre[0]), float(centre[1])))

    # -- bed polygon -------------------------------------------------------

    def bed_polygon_table(self, margin_in: float = 0.0) -> np.ndarray:
        m = float(margin_in)
        L, W = self.length_in, self.width_in
        return np.array(
            [[m, m], [L - m, m], [L - m, W - m], [m, W - m]], dtype=np.float64
        )

    def bed_polygon_image(self, margin_in: float = 0.0) -> np.ndarray:
        return self.table_to_image(self.bed_polygon_table(margin_in))

    def bed_mask(self, shape: Tuple[int, int], margin_in: float = 0.0) -> np.ndarray:
        """Binary mask (uint8 0/255) of the playing surface in image space."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = self.bed_polygon_image(margin_in).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask

    def contains(self, table_pt: Point, margin_in: float = 0.0) -> bool:
        x, y = table_pt
        m = margin_in
        return (m <= x <= self.length_in - m) and (m <= y <= self.width_in - m)

    # -- pockets -----------------------------------------------------------

    def pockets_table(self) -> np.ndarray:
        """Six pocket centres in table inches (empty for carom tables)."""
        if not self.has_pockets:
            return np.empty((0, 2), dtype=np.float64)
        L, W = self.length_in, self.width_in
        return np.array(
            [
                [0.0, 0.0],
                [L / 2.0, 0.0],
                [L, 0.0],
                [L, W],
                [L / 2.0, W],
                [0.0, W],
            ],
            dtype=np.float64,
        )

    def nearest_pocket_distance(self, table_pt: Point) -> float:
        pockets = self.pockets_table()
        if pockets.size == 0:
            return float("inf")
        d = np.linalg.norm(pockets - np.asarray(table_pt, dtype=np.float64), axis=1)
        return float(d.min())

    # -- rectification -----------------------------------------------------

    def overhead_homography(self, px_per_inch: float) -> np.ndarray:
        scale = np.array(
            [[px_per_inch, 0, 0], [0, px_per_inch, 0], [0, 0, 1]], dtype=np.float64
        )
        return scale @ self.H

    def warp_overhead(self, frame: np.ndarray, px_per_inch: float) -> np.ndarray:
        size = (
            int(round(self.length_in * px_per_inch)),
            int(round(self.width_in * px_per_inch)),
        )
        return cv2.warpPerspective(frame, self.overhead_homography(px_per_inch), size)

    # -- stability ---------------------------------------------------------

    def corner_drift(self, other: "TableModel") -> float:
        """Max corner displacement between two calibrations, in pixels."""
        return float(
            np.max(np.linalg.norm(self.corners_image - other.corners_image, axis=1))
        )

    def to_dict(self) -> dict:
        return {
            "corners_image": self.corners_image.tolist(),
            "length_in": self.length_in,
            "width_in": self.width_in,
            "ball_diameter_in": self.ball_diameter_in,
            "has_pockets": self.has_pockets,
            "mean_px_per_inch": self.mean_px_per_inch(),
            "ball_radius_px_at_centre": self.expected_ball_radius_px(
                tuple(self.corners_image.mean(axis=0))
            ),
        }
