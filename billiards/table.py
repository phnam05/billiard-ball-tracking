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
    sample_fraction: float = 0.0

    @property
    def _val_low(self) -> float:
        return self.val_halfwidth if self.val_halfwidth_low is None else self.val_halfwidth_low

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
        v_lo = max(0, int(self.val - self._val_low))
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


def _robust_halfwidth(values: np.ndarray, centre: float, sigmas: float, floor: float) -> float:
    """MAD-based half-width.  MAD is immune to the tail of ball pixels that
    inevitably leak into the sample, which a plain standard deviation is not."""
    if values.size == 0:
        return floor
    mad = float(np.median(np.abs(values.astype(np.float64) - centre)))
    sigma = 1.4826 * mad
    return float(max(floor, sigmas * sigma))


def estimate_cloth_color(frames: Sequence[np.ndarray], cfg: Config) -> ClothModel:
    """Measure the cloth colour from a set of BGR frames.

    Strategy: pool pixels from every sampled frame, discard anything too dark or
    too grey to be cloth, find the **joint** (H,S) peak, then measure robust
    spreads around it.  Hue and saturation get tight windows because they barely
    move with lighting; value gets a loose one because it moves a lot.
    """
    if not frames:
        raise ValueError("estimate_cloth_color needs at least one frame")

    ccfg = cfg.cloth
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

    h_cat = np.concatenate(h_all)
    s_cat = np.concatenate(s_all)
    v_cat = np.concatenate(v_all)

    hue = _circular_mode_hue(h_cat)

    # Restrict to the hue peak, then measure S and V on those pixels only.
    dh = np.abs(h_cat.astype(np.float64) - hue) % 180.0
    dh = np.minimum(dh, 180.0 - dh)
    near_hue = dh <= 12.0
    if np.count_nonzero(near_hue) < 64:
        near_hue = dh <= 25.0
    s_near = s_cat[near_hue]
    v_near = v_cat[near_hue]
    if s_near.size == 0:
        s_near, v_near = s_cat, v_cat

    sat = float(np.median(s_near))
    val = float(np.median(v_near))

    hue_hw = _robust_halfwidth(h_cat[near_hue].astype(np.float64), hue,
                               ccfg.hue_sigmas, ccfg.min_hue_halfwidth)
    hue_hw = float(min(hue_hw, 45.0))  # never let the hue window swallow the wheel
    sat_hw = _robust_halfwidth(s_near, sat, ccfg.sat_sigmas, ccfg.min_sat_halfwidth)
    val_hw = _robust_halfwidth(v_near, val, ccfg.val_sigmas, ccfg.min_val_halfwidth)

    model = ClothModel(
        hue=hue, sat=sat, val=val,
        hue_halfwidth=hue_hw, sat_halfwidth=sat_hw, val_halfwidth=val_hw,
        val_halfwidth_low=val_hw * ccfg.shadow_value_factor,
    )

    # Record how much of the frame the model claims, as a sanity signal.
    probe = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2HSV)
    mask = model.mask(probe)
    model.sample_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    return model


# --------------------------------------------------------------------------
# Table detection
# --------------------------------------------------------------------------


def largest_cloth_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest cloth blob, cleaned up.

    The closing kernel is scaled to the frame so it fills pocket jaws, ball
    shadows and the balls themselves (which are holes in the cloth) without
    being so large it eats the table edge.
    """
    h, w = mask.shape[:2]
    k = max(3, int(round(min(h, w) * 0.02)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # CHAIN_APPROX_NONE, not SIMPLE: SIMPLE compresses a straight run of
    # boundary pixels down to its two endpoints, which leaves the cushion line
    # fit in geometry.py with almost no points to work with.  We want every
    # boundary pixel here precisely so that fit is well conditioned.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.03 * h * w:
        return None
    return best


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

    table = TableModel(
        corners_image=corners,
        length_in=cfg.table.length_in,
        width_in=cfg.table.width_in,
        ball_diameter_in=cfg.table.ball_diameter_in,
        has_pockets=not cfg.table.preset.startswith("carom"),
    )
    return CalibrationResult(
        table=table,
        cloth=cloth,
        frames_used=len(quads),
        frames_attempted=len(frames),
        corner_spread_px=spread,
    )


def table_from_corners(corners: Sequence[Sequence[float]], cfg: Config) -> TableModel:
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
    )
