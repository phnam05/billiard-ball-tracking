"""Automatic cloth-colour estimation and table calibration.

Nothing in here is hand-tuned to a particular video.  The cloth colour is
*measured* from the clip, the table corners are *fitted*, and both are made
robust by sampling many frames and taking a median.

Why the old approach broke
--------------------------
``GetClothColor`` took the argmax of three **independent** 1-D histograms (H, S
and V separately) and put a fixed +-30 box around them.  The joint mode of a
colour distribution is not the product of its marginal modes, so on any frame
where, say, the players' shirts dominate the saturation histogram, the returned
"cloth colour" was a colour that appears nowhere in the image.  It also could
not represent red cloth at all, because hue wraps at 180 and a symmetric box
around h=2 or h=178 clips half the distribution away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .geometry import TableModel, quad_from_contour


# --------------------------------------------------------------------------
# Cloth colour
# --------------------------------------------------------------------------


@dataclass
class ClothModel:
    """A measured cloth colour with circular hue support."""

    hue: float
    sat: float
    val: float
    hue_halfwidth: float
    sat_halfwidth: float
    val_halfwidth: float
    #: Downward half-width, which is larger than the upward one: see
    #: ClothConfig.shadow_value_factor.  Defaults to val_halfwidth when unset.
    val_halfwidth_low: Optional[float] = None
    #: Multiplicative floor on the dark side of the value window.
    min_value_ratio: float = 0.55
    sample_fraction: float = 0.0
    #: A grey (or black, or white) cloth: too little colour for a hue, so it
    #: is told apart by being unsaturated and about this bright instead.  See
    #: ``estimate_cloth_color(..., neutral_ok=True)``.
    neutral: bool = False

    @property
    def _val_low(self) -> float:
        return self.val_halfwidth if self.val_halfwidth_low is None else self.val_halfwidth_low

    @property
    def value_low(self) -> float:
        """Darkest value still counted as cloth."""
        return max(self.val - self._val_low, self.min_value_ratio * self.val)

    def hue_lut(self) -> np.ndarray:
        """256-entry LUT marking hues that count as cloth, wrapping at 180."""
        hues = np.arange(256, dtype=np.float32)
        delta = np.abs(hues - self.hue) % 180.0
        delta = np.minimum(delta, 180.0 - delta)
        lut = (delta <= self.hue_halfwidth).astype(np.uint8) * 255
        lut[180:] = 0  # OpenCV hue is 0..179; anything above is not a real hue
        return lut

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary cloth mask (uint8 0/255) for an HSV image."""
        h, s, v = cv2.split(hsv)
        s_lo = max(0, int(self.sat - self.sat_halfwidth))
        s_hi = min(255, int(self.sat + self.sat_halfwidth))
        if self.neutral:
            # Hue is noise on a colourless cloth; being unsaturated and about
            # this bright is what makes it cloth.
            v_lo = max(0, int(self.value_low))
            v_hi = min(255, int(self.val + self.val_halfwidth))
            s_ok = cv2.inRange(s, np.array(0, np.uint8), np.array(s_hi, np.uint8))
            v_ok = cv2.inRange(v, np.array(v_lo, np.uint8), np.array(v_hi, np.uint8))
            return cv2.bitwise_and(s_ok, v_ok)
        h_ok = cv2.LUT(h, self.hue_lut())
        v_lo = max(0, int(self.value_low))
        v_hi = min(255, int(self.val + self.val_halfwidth))
        s_ok = cv2.inRange(s, np.array(s_lo, np.uint8), np.array(s_hi, np.uint8))
        v_ok = cv2.inRange(v, np.array(v_lo, np.uint8), np.array(v_hi, np.uint8))
        return cv2.bitwise_and(h_ok, cv2.bitwise_and(s_ok, v_ok))

    def to_dict(self) -> dict:
        return {
            "hue": round(self.hue, 2),
            "sat": round(self.sat, 2),
            "val": round(self.val, 2),
            "hue_halfwidth": round(self.hue_halfwidth, 2),
            "sat_halfwidth": round(self.sat_halfwidth, 2),
            "val_halfwidth": round(self.val_halfwidth, 2),
            "val_halfwidth_low": round(self._val_low, 2),
            "value_low": round(self.value_low, 2),
            "sample_fraction": round(self.sample_fraction, 4),
            "neutral": self.neutral,
        }


def _circular_mode_hue(hues: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Hue histogram peak, refined by a circular mean over the peak's neighbours."""
    hist = np.bincount(hues.astype(np.int32).ravel(), weights=weights, minlength=180)[:180]
    hist = hist.astype(np.float64)
    # Smooth circularly so a bimodal-by-noise histogram still gives one peak.
    kernel = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    peak = int(np.argmax(smoothed))

    # Circular mean over +-6 bins around the peak for sub-bin accuracy.
    idx = (np.arange(peak - 6, peak + 7)) % 180
    w = smoothed[idx]
    if w.sum() <= 0:
        return float(peak)
    angles = np.deg2rad(idx.astype(np.float64) * 2.0)  # 0..179 covers 0..358 deg
    mean_angle = np.arctan2(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles)))
    return float((np.rad2deg(mean_angle) / 2.0) % 180.0)


def _robust_halfwidth(
    values: np.ndarray, centre: float, sigmas: float, floor: float, ceiling: float
) -> float:
    """MAD-based half-width, clamped.

    MAD is immune to the tail of ball pixels that inevitably leak into the
    sample, which a plain standard deviation is not.  The ceiling matters just
    as much: if the background shares the cloth's hue, even a robust spread can
    come out so wide that the resulting window accepts the whole image.
    """
    if values.size == 0:
        return floor
    mad = float(np.median(np.abs(values.astype(np.float64) - centre)))
    sigma = 1.4826 * mad
    return float(np.clip(sigmas * sigma, floor, ceiling))


def _model_from_pixels(
    h: np.ndarray, s: np.ndarray, v: np.ndarray, cfg: Config
) -> ClothModel:
    """Fit a cloth colour window to a pool of HSV samples."""
    ccfg = cfg.cloth
    hue = _circular_mode_hue(h)

    dh = np.abs(h.astype(np.float64) - hue) % 180.0
    dh = np.minimum(dh, 180.0 - dh)
    near_hue = dh <= 12.0
    if np.count_nonzero(near_hue) < 64:
        near_hue = dh <= 25.0
    if not np.any(near_hue):
        near_hue = np.ones_like(dh, dtype=bool)

    s_near = s[near_hue]
    v_near = v[near_hue]
    sat = float(np.median(s_near))
    val = float(np.median(v_near))

    hue_hw = _robust_halfwidth(
        h[near_hue].astype(np.float64), hue,
        ccfg.hue_sigmas, ccfg.min_hue_halfwidth, ccfg.max_hue_halfwidth,
    )
    sat_hw = _robust_halfwidth(
        s_near, sat, ccfg.sat_sigmas, ccfg.min_sat_halfwidth, ccfg.max_sat_halfwidth
    )
    val_hw = _robust_halfwidth(
        v_near, val, ccfg.val_sigmas, ccfg.min_val_halfwidth, ccfg.max_val_halfwidth
    )

    return ClothModel(
        hue=hue, sat=sat, val=val,
        hue_halfwidth=hue_hw, sat_halfwidth=sat_hw, val_halfwidth=val_hw,
        val_halfwidth_low=val_hw * ccfg.shadow_value_factor,
        min_value_ratio=ccfg.shadow_min_value_ratio,
    )


def _neutral_model(
    s: np.ndarray, v: np.ndarray, frames: Sequence[np.ndarray], cfg: Config
) -> ClothModel:
    """A grey cloth: how unsaturated and how bright it is, with no hue."""
    ccfg = cfg.cloth
    sat = float(np.median(s))
    val = float(np.median(v))
    # Tighter than a coloured cloth's saturation window: on grey cloth the
    # balls are told apart mostly by having colour at all, and a dull one --
    # the maroon 7 -- is not far above the cloth.
    sat_hw = _robust_halfwidth(s, sat, ccfg.sat_sigmas, 18.0, ccfg.max_sat_halfwidth)
    # Brightness is all that separates the cloth from the cue ball, which is
    # unsaturated too, so the window above the cloth is kept narrow: lit
    # cloth is barely brighter than its median.  Below, shadows still need the
    # usual room.
    val_hw = _robust_halfwidth(v, val, ccfg.val_sigmas, ccfg.min_val_halfwidth, ccfg.max_val_halfwidth)
    val_hw_up = _robust_halfwidth(v, val, 3.5, 24.0, ccfg.max_val_halfwidth)
    model = ClothModel(
        hue=0.0, sat=sat, val=val, hue_halfwidth=180.0, sat_halfwidth=sat_hw,
        val_halfwidth=val_hw_up, val_halfwidth_low=val_hw * ccfg.shadow_value_factor,
        min_value_ratio=ccfg.shadow_min_value_ratio, neutral=True,
    )
    probe = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2HSV)
    model.sample_fraction = float(np.count_nonzero(model.mask(probe))) / float(probe[:, :, 0].size)
    return model


def _pixels_in_dominant_region(
    hsvs: Sequence[np.ndarray], model: ClothModel
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collect HSV samples from inside the biggest region the model selects.

    This is the refinement step.  The first estimate is made over the whole
    frame and is therefore pulled towards whatever else shares the cloth's hue
    -- in tournament footage, the blue banners behind a blue table.  Re-measuring
    using only pixels well inside the largest selected region converges the
    estimate onto the bed, and the window tightens accordingly.
    """
    hs: List[np.ndarray] = []
    ss: List[np.ndarray] = []
    vs: List[np.ndarray] = []

    for hsv in hsvs:
        mask = model.mask(hsv)
        h, w = mask.shape[:2]
        k = max(3, int(round(min(h, w) * 0.012)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        if n <= 1:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] < 0.02 * h * w:
            continue

        region = (labels == biggest).astype(np.uint8) * 255
        # Erode well inside: the boundary is contaminated by rails and cushions.
        erode_k = max(3, int(round(min(h, w) * 0.03)) | 1)
        region = cv2.erode(
            region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        )
        inside = region > 0
        if np.count_nonzero(inside) < 500:
            continue

        hs.append(hsv[:, :, 0][inside])
        ss.append(hsv[:, :, 1][inside])
        vs.append(hsv[:, :, 2][inside])

    if not hs:
        return None
    return np.concatenate(hs), np.concatenate(ss), np.concatenate(vs)


def estimate_cloth_color(
    frames: Sequence[np.ndarray], cfg: Config, neutral_ok: bool = False
) -> ClothModel:
    """Measure the cloth colour from a set of BGR frames.

    Pool pixels from every sampled frame, discard anything too dark or too grey
    to be cloth, find the hue mode on a circular histogram, then measure robust
    spreads around it.  Hue and saturation get tight windows because they barely
    move with lighting; value gets a loose, asymmetric one because it moves a
    lot and only ever downwards (shadow).

    The estimate is then refined by re-measuring inside the region it selected,
    which is what stops a background that happens to share the cloth's hue from
    widening the window until it accepts everything.

    With ``neutral_ok`` the frames show only the bed (everything else blacked
    out, because the corners are known), and if most of it has too little
    colour to count, the cloth is grey: it is then modelled by its low
    saturation and its brightness, with no hue.  Searched for over a whole
    frame, a grey cloth is indistinguishable from a grey floor, which is why
    this is only done inside known corners.
    """
    if not frames:
        raise ValueError("estimate_cloth_color needs at least one frame")

    ccfg = cfg.cloth
    hsvs: List[np.ndarray] = []
    h_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []
    v_all: List[np.ndarray] = []

    lit_s: List[np.ndarray] = []
    lit_v: List[np.ndarray] = []
    for frame in frames:
        small = frame
        if ccfg.analysis_scale != 1.0:
            small = cv2.resize(
                frame, None, fx=ccfg.analysis_scale, fy=ccfg.analysis_scale,
                interpolation=cv2.INTER_AREA,
            )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsvs.append(hsv)
        h, s, v = cv2.split(hsv)
        lit = v >= ccfg.min_value
        if neutral_ok:
            lit_s.append(s[lit])
            lit_v.append(v[lit])
        keep = lit & (s >= ccfg.min_saturation)
        if not np.any(keep):
            continue
        h_all.append(h[keep])
        s_all.append(s[keep])
        v_all.append(v[keep])

    if neutral_ok and lit_s:
        s_pool, v_pool = np.concatenate(lit_s), np.concatenate(lit_v)
        coloured = sum(len(x) for x in h_all)
        if s_pool.size and coloured < 0.5 * s_pool.size:
            return _neutral_model(s_pool, v_pool, frames, cfg)

    if not h_all:
        raise RuntimeError(
            "Could not measure the cloth colour: every pixel was rejected as too "
            "dark or too grey. If the clip really is very dark, lower "
            "cloth.min_value / cloth.min_saturation."
        )

    model = _model_from_pixels(
        np.concatenate(h_all), np.concatenate(s_all), np.concatenate(v_all), cfg
    )

    for _ in range(max(0, ccfg.refine_iterations)):
        sample = _pixels_in_dominant_region(hsvs, model)
        if sample is None:
            break
        refined = _model_from_pixels(sample[0], sample[1], sample[2], cfg)
        # Keep the refinement only if it actually narrows the selection; a
        # refinement that grows the window is a sign the region was wrong.
        model = refined

    # Record how much of the frame the model claims, as a sanity signal.
    probe = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2HSV)
    mask = model.mask(probe)
    model.sample_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    return model


# --------------------------------------------------------------------------
# Table detection
# --------------------------------------------------------------------------


def largest_cloth_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Outline of the table bed, isolated from anything else the mask caught.

    Order of operations matters here, and getting it wrong is what made this
    fail on real broadcast footage.  Opening runs **first**, to sever the thin
    bridges that connect the bed to same-coloured background -- blue banners
    behind a blue table, a grey wall behind grey cloth.  Only then is the
    largest component chosen; closing is applied to that component alone, to
    fill in the balls and pocket jaws, which are holes in the cloth.

    Closing first (as an earlier version did) welds the bed to the background
    before anything gets to separate them, and the "table" then spans half the
    frame.
    """
    h, w = mask.shape[:2]
    open_k = max(3, int(round(min(h, w) * 0.014)) | 1)
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.03 * h * w:
        return None

    region = (labels == biggest).astype(np.uint8) * 255
    close_k = max(3, int(round(min(h, w) * 0.03)) | 1)
    region = cv2.morphologyEx(
        region, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )

    # CHAIN_APPROX_NONE, not SIMPLE: SIMPLE compresses a straight run of
    # boundary pixels down to its two endpoints, which leaves the cushion line
    # fit in geometry.py with almost no points to work with.  We want every
    # boundary pixel here precisely so that fit is well conditioned.
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------------------------------
# From the cloth's outline to the cushion noses
# --------------------------------------------------------------------------

#: Weights of L, a, b in the colour step that marks a nose -- as elsewhere,
#: lightness counts for less.
_NOSE_LAB_WEIGHTS = np.array([0.45, 1.0, 1.0])
#: How far inside the fitted edge the rays start, and their step, in inches.
_NOSE_STEP_IN = 0.1
#: A rail on which fewer rays than this found an edge is left where it is.
_NOSE_MIN_AGREEING = 0.4


def refine_to_cushion_noses(
    table: TableModel, frames: Sequence[np.ndarray], cfg: Config
) -> Tuple[TableModel, dict]:
    """Move each fitted edge in to the nose of its cushion.

    The calibrated outline is the outline of the *cloth*, and a table's
    cushions are clothed too.  Lit from above, a cushion's top looks just like
    the bed, so on the sample broadcasts the fitted long rails ran out over
    the cushion tops, about two inches outside the noses, and the far rail
    over the far cushion's face.  The table was then fitted 4 in too long and
    wide, every position near those rails was off, and balls bouncing off
    them turned round 3.3 in (long rails) and 5.7 in (far rail) from the rail
    instead of the ball radius.

    What marks a nose is its face: undercut and in shadow, a dark line seen
    from the side and a dark band seen from in front.  So along each rail,
    rays are walked from well inside the bed outwards, over a median of
    several frames (so that balls and players drop out), and the edge is
    where the colour first leaves the bed's -- at half height of the step, so
    a thin line and a broad face are placed alike.

    The step is where the nose appears in the picture.  On a rail whose face
    the camera sees, that is the foot of the face, on the cloth.  On the rail
    nearest the camera it is the nose itself, a ball's 63% above the cloth,
    whose image lands on the cloth well beyond it -- 2.8 in on the sample
    broadcasts -- so that rail is placed through the plane at nose height.

    Returns the refined table (with the original outline kept as
    ``outline_image``, which is what a later outline is compared with) and
    what was found per rail, in inches inward.
    """
    tc = cfg.table
    report: dict = {}
    if not tc.fit_cushion_noses or not frames:
        return table, report

    lab = np.median(
        np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2Lab) for f in frames[:9]]), axis=0
    ).astype(np.float32)
    h, w = lab.shape[:2]
    L, W = table.length_in, table.width_in

    def sample(points_table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = np.rint(table.table_to_image(points_table)).astype(np.int64)
        ok = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        out = np.zeros((len(points_table), 3), np.float32)
        out[ok] = lab[img[ok, 1], img[ok, 0]]
        return out, ok

    grid = np.array([[x, y] for x in np.linspace(0.15 * L, 0.85 * L, 24)
                     for y in np.linspace(0.25 * W, 0.75 * W, 10)])
    bed_px, ok = sample(grid)
    if np.count_nonzero(ok) < 20:
        return table, report
    bed = np.median(bed_px[ok], axis=0)
    noise = float(np.percentile(np.linalg.norm((bed_px[ok] - bed) * _NOSE_LAB_WEIGHTS, axis=1), 90))
    step_min = max(tc.cushion_nose_min_step, 3.0 * noise)

    depth_max = tc.cushion_nose_search_in
    depths = np.arange(depth_max, -_NOSE_STEP_IN / 2, -_NOSE_STEP_IN)  # inside -> edge
    camera = None if table.camera is None else np.asarray(table.camera["position_in"])
    nose_h = tc.cushion_nose_height_ball_diameters * table.ball_diameter_in
    raised = None
    if camera is not None and table.image_size is not None:
        from .geometry import raised_plane_homography

        found = raised_plane_homography(
            table.H_inv, table.image_to_table(table.corners_image), table.image_size, nose_h
        )
        if found is not None:
            raised = np.linalg.inv(found[0])  # image -> table, at nose height

    # (name, point on the rail at (along, depth), rail length, camera inside?)
    rails = [
        ("y=0", lambda u, d: (u, d), L, camera is None or camera[1] > 0.0),
        ("y=W", lambda u, d: (u, W - d), L, camera is None or camera[1] < W),
        ("x=0", lambda u, d: (d, u), W, camera is None or camera[0] > 0.0),
        ("x=L", lambda u, d: (L - d, u), W, camera is None or camera[0] < L),
    ]
    offsets = {}
    for name, at, length, sees_face in rails:
        # Clear of the pockets, which bite the rail at both ends and, on a
        # long rail, in the middle.
        clear = 3.5 * table.ball_diameter_in
        us = [u for u in np.linspace(clear, length - clear, 48)
              if length < 1.5 * W or abs(u - length / 2.0) > clear]
        hits = []
        for u in us:
            pts = np.array([at(u, d) for d in depths])
            px, ok = sample(pts)
            if not np.all(ok):
                continue
            dist = np.linalg.norm((px - bed) * _NOSE_LAB_WEIGHTS, axis=1)
            over = np.nonzero(dist > step_min)[0]
            if over.size == 0:
                continue
            first = int(over[0])
            if first < 3:  # the bed is not bed-coloured this far in: no read
                continue
            # Half height of the step, between the bed and the peak just past it.
            base = float(np.median(dist[:first]))
            peak = float(np.max(dist[first:first + int(round(1.0 / _NOSE_STEP_IN)) + 1]))
            half = 0.5 * (base + peak)
            k = first
            while k > 0 and dist[k - 1] >= half:
                k -= 1
            lo, hi = dist[k - 1], dist[k]
            frac = 0.0 if hi <= lo else float(np.clip((half - lo) / (hi - lo), 0.0, 1.0))
            hits.append(depths[k - 1] - frac * _NOSE_STEP_IN if k > 0 else depths[0])
        if len(hits) < _NOSE_MIN_AGREEING * len(us):
            report[name] = {"inches": 0.0, "rays": len(hits), "of": len(us), "moved": False}
            continue
        d = float(np.median(hits))
        if not sees_face and raised is not None:
            # The step is the nose seen from above and behind; find where
            # the nose is, not where its image lands on the cloth.
            img = table.table_to_image(np.array([at(length / 2.0, d)]))
            nose = cv2.perspectiveTransform(img.reshape(-1, 1, 2), raised).reshape(2)
            rail0 = np.array(at(length / 2.0, 0.0))
            inward = np.array(at(length / 2.0, 1.0)) - rail0
            d = float(np.dot(nose - rail0, inward))
        d = float(np.clip(d, 0.0, depth_max))
        spread = float(np.percentile(hits, 75) - np.percentile(hits, 25))
        offsets[name] = d
        report[name] = {"inches": round(d, 2), "rays": len(hits), "of": len(us),
                        "spread_in": round(spread, 2), "moved": d > 0.0}

    if not offsets:
        return table, report
    x0, x1 = offsets.get("x=0", 0.0), L - offsets.get("x=L", 0.0)
    y0, y1 = offsets.get("y=0", 0.0), W - offsets.get("y=W", 0.0)
    corners = table.table_to_image(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    from .geometry import order_corners

    refined = TableModel(
        corners_image=order_corners(corners),
        length_in=table.length_in,
        width_in=table.width_in,
        ball_diameter_in=table.ball_diameter_in,
        has_pockets=table.has_pockets,
        image_size=table.image_size,
        ball_parallax=table.ball_parallax,
        outline_image=table.outline_image if table.outline_image is not None else table.corners_image,
    )
    return refined, report


# --------------------------------------------------------------------------
# Which colour is the cloth
# --------------------------------------------------------------------------
#
# ``estimate_cloth_color`` takes the commonest saturated hue for the cloth.
# That holds at a venue with grey carpet, and it is what every sample clip
# was.  It fails wherever something else of a strong colour fills more of
# the picture than the bed does: the 2026 US Open's royal-blue floor (hue
# 117, saturation 211) around a blue-grey cloth (hue 103, saturation 42), a
# red floor under green cloth.  There the "cloth" was the floor, and nothing
# was tracked.  So the commonest colours -- by hue *and* saturation, and the
# commonest greys -- are each tried as the cloth, and the one whose outline
# behaves like a table is kept.


@dataclass
class ClothCandidate:
    """One colour tried as the cloth, and how table-like its outline was."""

    source: str
    model: ClothModel
    quads: List[np.ndarray]
    frames: int
    #: The outlines of the biggest group of frames that agree with each
    #: other: one camera angle, where a highlight reel has several.
    view: List[np.ndarray] = field(default_factory=list)
    #: How many frames that is.
    inliers: int = 0
    #: How much of the median outline the colour fills (a bed: most of it; a
    #: floor, whose outline encloses the table: much less).
    fill: float = 0.0
    #: Corners of the median outline on the edge of the picture.  A floor
    #: runs off the picture; a table that does cannot be calibrated anyway.
    border_corners: int = 0
    ball_radius_px: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "hue": round(self.model.hue, 1),
            "sat": round(self.model.sat, 1),
            "val": round(self.model.val, 1),
            "neutral": self.model.neutral,
            "outlines": len(self.quads),
            "agreeing": self.inliers,
            "of": self.frames,
            "fill": round(self.fill, 3),
            "border_corners": self.border_corners,
            "ball_radius_px": round(self.ball_radius_px, 2),
            "score": round(self.score, 4),
        }


def _refined(model: ClothModel, hsvs: Sequence[np.ndarray], frames: Sequence[np.ndarray],
             cfg: Config) -> ClothModel:
    """Re-measure a candidate inside the region it selects, as the main estimate is."""
    for _ in range(max(0, cfg.cloth.refine_iterations)):
        sample = _pixels_in_dominant_region(hsvs, model)
        if sample is None:
            break
        if model.neutral:
            model = _neutral_model(sample[1], sample[2], frames, cfg)
        else:
            model = _model_from_pixels(sample[0], sample[1], sample[2], cfg)
    return model


def _peaks(hist: np.ndarray, count: int, floor: float, reach: Tuple[int, ...],
           wrap_first_axis: bool) -> List[Tuple[int, ...]]:
    """The ``count`` biggest bins of a histogram at least ``floor`` high, each
    blanking its neighbourhood (``reach`` bins either way) before the next."""
    work = hist.astype(np.float64).copy()
    found: List[Tuple[int, ...]] = []
    for _ in range(count):
        idx = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[idx] < floor:
            break
        found.append(tuple(int(i) for i in idx))
        ranges = []
        for axis, (i, r) in enumerate(zip(idx, reach)):
            span = np.arange(i - r, i + r + 1)
            if axis == 0 and wrap_first_axis:
                span %= work.shape[0]
            else:
                span = span[(span >= 0) & (span < work.shape[axis])]
            ranges.append(span)
        work[np.ix_(*ranges)] = 0.0
    return found


def cloth_candidates(frames: Sequence[np.ndarray], cfg: Config) -> List[Tuple[str, ClothModel]]:
    """Colours that could be the cloth: the usual estimate first, then the
    commonest colours by hue and saturation, then the commonest greys."""
    ccfg = cfg.cloth
    out: List[Tuple[str, ClothModel]] = []
    try:
        out.append(("the commonest colour", estimate_cloth_color(frames, cfg)))
    except RuntimeError:
        pass  # every pixel too grey: only the grey candidates below can be it

    hsvs = []
    for frame in frames:
        small = frame
        if ccfg.analysis_scale != 1.0:
            small = cv2.resize(frame, None, fx=ccfg.analysis_scale, fy=ccfg.analysis_scale,
                               interpolation=cv2.INTER_AREA)
        hsvs.append(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    pix = np.concatenate([hsv.reshape(-1, 3) for hsv in hsvs])
    h, s, v = pix[:, 0].astype(np.int32), pix[:, 1].astype(np.int32), pix[:, 2].astype(np.int32)
    lit = v >= ccfg.min_value
    floor = ccfg.candidate_min_share * pix.shape[0]

    # Colours: a joint hue x saturation histogram, 4 hue units by 16 levels.
    coloured = lit & (s >= ccfg.min_saturation)
    hb, sb = np.minimum(h[coloured] // 4, 44), s[coloured] // 16
    hist = np.bincount(hb * 16 + sb, minlength=45 * 16).reshape(45, 16)
    for hi, si in _peaks(hist, ccfg.colour_candidates, floor, (2, 2), wrap_first_axis=True):
        hue, sat = hi * 4 + 2.0, si * 16 + 8.0
        dh = np.abs(h - hue) % 180
        near = coloured & (np.minimum(dh, 180 - dh) <= 8) & (np.abs(s - sat) <= 32)
        if np.count_nonzero(near) < 500:
            continue
        model = _model_from_pixels(h[near], s[near], v[near], cfg)
        out.append((f"colour at hue {hue:.0f}, saturation {sat:.0f}", _refined(model, hsvs, frames, cfg)))

    # Greys: by brightness alone, 8 levels a bin.
    grey = lit & (s < ccfg.min_saturation)
    ghist = np.bincount(v[grey] // 8, minlength=32)
    for (vi,) in _peaks(ghist, ccfg.grey_candidates, floor, (3,), wrap_first_axis=False):
        val = vi * 8 + 4.0
        near = grey & (np.abs(v - val) <= 20)
        if np.count_nonzero(near) < 500:
            continue
        model = _neutral_model(s[near], v[near], frames, cfg)
        out.append((f"grey at brightness {val:.0f}", _refined(model, hsvs, frames, cfg)))
    return out


def corners_on_edge(quad: np.ndarray, width: int, height: int, margin: float = 2.5) -> int:
    """How many corners of an outline lie on the edge of the picture.  Two or
    more and it runs off the picture: a floor, a close-up, not a table whose
    corners can be seen."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return int(np.count_nonzero(
        (q[:, 0] <= margin) | (q[:, 0] >= width - 1 - margin)
        | (q[:, 1] <= margin) | (q[:, 1] >= height - 1 - margin)
    ))


def _outline_fill(mask: np.ndarray, quad: np.ndarray) -> float:
    inside = np.zeros_like(mask)
    cv2.fillConvexPoly(inside, np.round(quad).astype(np.int32), 255)
    area = cv2.countNonZero(inside)
    return float(cv2.countNonZero(cv2.bitwise_and(mask, inside))) / area if area else 0.0


def score_candidate(source: str, model: ClothModel, frames: Sequence[np.ndarray],
                    cfg: Config) -> ClothCandidate:
    """How much the outline of this colour behaves like a table's.

    ``score = agreeing frames / frames x fill x 0.5 ** corners on the edge``,
    and 0 with two or more corners on the edge, or if a ball on it would be
    too small to see.  "Agreeing" is the biggest group of frames whose
    outlines agree with each other, not agreement with the median of them
    all: a highlight reel cuts between two or three cameras, and a median of
    their outlines is an outline no camera saw.  Each factor is one thing
    a floor or a banner does and a bed does not: its outline jumps from frame
    to frame, it encloses things of another colour (the table), it runs off
    the picture.
    """
    cand = ClothCandidate(source=source, model=model, quads=[], frames=len(frames))
    fills: List[float] = []
    for frame in frames:
        mask = model.mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        contour = largest_cloth_contour(mask)
        quad = quad_from_contour(contour) if contour is not None else None
        if quad is None:
            continue
        cand.quads.append(quad)
        fills.append(_outline_fill(mask, quad))
    if len(cand.quads) < 3:
        return cand
    stack = np.stack(cand.quads)
    h, w = frames[0].shape[:2]
    tol = cfg.table.candidate_corner_tolerance * float(np.hypot(w, h))
    # The outline most others agree with seeds the view; the view is then
    # every outline close to the median of that group.
    pairwise = np.linalg.norm(stack[:, None] - stack[None], axis=3).max(axis=2)
    seed = int(np.argmax((pairwise <= tol).sum(axis=1)))
    median = np.median(stack[pairwise[seed] <= tol], axis=0)
    agree = np.linalg.norm(stack - median, axis=2).max(axis=1) <= tol
    if np.count_nonzero(agree) >= 3:
        median = np.median(stack[agree], axis=0)
    cand.view = [q for q, a in zip(cand.quads, agree) if a]
    cand.inliers = len(cand.view)
    cand.fill = float(np.median(np.asarray(fills)[agree])) if cand.inliers else 0.0
    cand.border_corners = corners_on_edge(median, w, h)
    if cand.border_corners >= 2:
        # Runs off the picture: a floor, or the whole frame.  A table whose
        # corners are out of view could not be calibrated from anyway.
        return cand
    try:
        table = TableModel(
            corners_image=median, length_in=cfg.table.length_in, width_in=cfg.table.width_in,
            ball_diameter_in=cfg.table.ball_diameter_in, image_size=(w, h),
            ball_parallax=cfg.table.ball_parallax,
        )
        cand.ball_radius_px = float(table.expected_ball_radius_px(tuple(median.mean(axis=0))))
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return cand
    if cand.ball_radius_px < cfg.table.min_ball_radius_px:
        return cand
    cand.score = cand.inliers / cand.frames * cand.fill * 0.5 ** cand.border_corners
    return cand


def choose_cloth(frames: Sequence[np.ndarray], cfg: Config) -> Tuple[ClothCandidate, List[ClothCandidate]]:
    """The candidate to calibrate with, and every one that was tried.

    The usual estimate is kept unless another is clearly more table-like
    (``cloth.candidate_margin``), so footage it already handles is not
    re-decided on a coin toss between two near-identical windows.
    """
    tried = [score_candidate(source, model, frames, cfg) for source, model in cloth_candidates(frames, cfg)]
    if not tried:
        raise RuntimeError(
            "Could not measure the cloth colour: every pixel was rejected as too "
            "dark. If the clip really is very dark, lower cloth.min_value."
        )
    best = tried[0]
    for cand in tried[1:]:
        if cand.score > best.score * (cfg.cloth.candidate_margin if best is tried[0] else 1.0):
            best = cand
    return best, tried


@dataclass
class CalibrationResult:
    table: TableModel
    cloth: ClothModel
    frames_used: int
    frames_attempted: int
    corner_spread_px: float
    #: Per rail, how far in from the cloth's outline its nose was found.
    cushion_noses: dict = field(default_factory=dict)
    #: Every colour tried as the cloth, and how table-like it was.
    cloth_candidates: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "table": self.table.to_dict(),
            "cloth": self.cloth.to_dict(),
            "frames_used": self.frames_used,
            "frames_attempted": self.frames_attempted,
            "corner_spread_px": round(self.corner_spread_px, 2),
            "cushion_noses": self.cushion_noses,
            "cloth_candidates": self.cloth_candidates,
        }


def calibrate(frames: Sequence[np.ndarray], cfg: Config) -> CalibrationResult:
    """Estimate cloth colour and table geometry from sampled frames.

    Per-corner median across frames is what makes this survive a player leaning
    over the rail, a cue crossing the cushion, or a caption bar: those spoil a
    minority of frames, and the median ignores them.
    """
    chosen, tried = choose_cloth(frames, cfg)
    cloth = chosen.model
    # One camera's view of the table, where the frames show several.
    quads = chosen.view if len(chosen.view) >= 3 else chosen.quads

    if len(quads) < 3:
        raise RuntimeError(
            "Table calibration failed: found the cloth in only "
            f"{len(quads)}/{len(frames)} sampled frames. "
            "Check that the whole table bed is visible, or pass "
            "--table-corners to set the four corners by hand."
        )

    stack = np.stack(quads, axis=0)  # (n, 4, 2)
    corners = np.median(stack, axis=0)
    spread = float(np.median(np.linalg.norm(stack - corners, axis=2).max(axis=1)))

    h, w = frames[0].shape[:2]
    table = TableModel(
        corners_image=corners,
        length_in=cfg.table.length_in,
        width_in=cfg.table.width_in,
        ball_diameter_in=cfg.table.ball_diameter_in,
        has_pockets=not cfg.table.preset.startswith("carom"),
        image_size=(w, h),
        ball_parallax=cfg.table.ball_parallax,
    )
    ball_r = table.expected_ball_radius_px(tuple(table.corners_image.mean(axis=0)))
    if ball_r < cfg.table.min_ball_radius_px:
        # Something cloth-coloured was found, but not a table anyone could
        # track: on grey cloth over a grey floor it was a blue banner, and
        # every ball would have been 6 px across.  Better to say so, and ask
        # for the corners, than to track nothing and look as if it worked.
        raise RuntimeError(
            "Table calibration failed: the cloth-coloured region found is too "
            f"small to be the table (a ball there would be {2 * ball_r:.1f} px "
            "across), so it is probably something else of that colour. Pass "
            "--table-corners to set the four corners by hand."
        )
    table, noses = refine_to_cushion_noses(table, frames, cfg)
    return CalibrationResult(
        table=table,
        cloth=cloth,
        frames_used=len(quads),
        frames_attempted=len(frames),
        corner_spread_px=spread,
        cushion_noses=noses,
        cloth_candidates=[dict(c.to_dict(), chosen=c is chosen) for c in tried],
    )


def table_from_corners(
    corners: Sequence[Sequence[float]],
    cfg: Config,
    image_size: Optional[Tuple[int, int]] = None,
) -> TableModel:
    """Build a TableModel from four manually supplied image corners."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("Exactly four table corners are required")
    from .geometry import order_corners

    return TableModel(
        corners_image=order_corners(pts),
        length_in=cfg.table.length_in,
        width_in=cfg.table.width_in,
        ball_diameter_in=cfg.table.ball_diameter_in,
        has_pockets=not cfg.table.preset.startswith("carom"),
        image_size=image_size,
        ball_parallax=cfg.table.ball_parallax,
    )
