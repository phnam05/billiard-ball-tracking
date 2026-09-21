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
        h_ok = cv2.LUT(h, self.hue_lut())
        s_lo = max(0, int(self.sat - self.sat_halfwidth))
        s_hi = min(255, int(self.sat + self.sat_halfwidth))
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


def estimate_cloth_color(frames: Sequence[np.ndarray], cfg: Config) -> ClothModel:
    """Measure the cloth colour from a set of BGR frames.

    Pool pixels from every sampled frame, discard anything too dark or too grey
    to be cloth, find the hue mode on a circular histogram, then measure robust
    spreads around it.  Hue and saturation get tight windows because they barely
    move with lighting; value gets a loose, asymmetric one because it moves a
    lot and only ever downwards (shadow).

    The estimate is then refined by re-measuring inside the region it selected,
    which is what stops a background that happens to share the cloth's hue from
    widening the window until it accepts everything.
    """
    if not frames:
        raise ValueError("estimate_cloth_color needs at least one frame")

    ccfg = cfg.cloth
    hsvs: List[np.ndarray] = []
    h_all: List[np.ndarray] = []
    s_all: List[np.ndarray] = []
    v_all: List[np.ndarray] = []

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
        keep = (v >= ccfg.min_value) & (s >= ccfg.min_saturation)
        if not np.any(keep):
            continue
        h_all.append(h[keep])
        s_all.append(s[keep])
        v_all.append(v[keep])

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


@dataclass
class CalibrationResult:
    table: TableModel
    cloth: ClothModel
    frames_used: int
    frames_attempted: int
    corner_spread_px: float

    def to_dict(self) -> dict:
        return {
            "table": self.table.to_dict(),
            "cloth": self.cloth.to_dict(),
            "frames_used": self.frames_used,
            "frames_attempted": self.frames_attempted,
            "corner_spread_px": round(self.corner_spread_px, 2),
        }


def calibrate(frames: Sequence[np.ndarray], cfg: Config) -> CalibrationResult:
    """Estimate cloth colour and table geometry from sampled frames.

    Per-corner median across frames is what makes this survive a player leaning
    over the rail, a cue crossing the cushion, or a caption bar: those spoil a
    minority of frames, and the median ignores them.
    """
    cloth = estimate_cloth_color(frames, cfg)

    quads: List[np.ndarray] = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cloth.mask(hsv)
        contour = largest_cloth_contour(mask)
        if contour is None:
            continue
        quad = quad_from_contour(contour)
        if quad is not None:
            quads.append(quad)

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
    )
    return CalibrationResult(
        table=table,
        cloth=cloth,
        frames_used=len(quads),
        frames_attempted=len(frames),
        corner_spread_px=spread,
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
    )
