"""What the set-up page shows: where the tracker thinks the table is.

Everything the tracker measures before the first frame -- the cloth colour,
the table's corners, the ball size -- is checked here on one frame the user
picks, with the detections it would make there, so a video can be put right
before an hour is spent tracking it.  When no table is found the reply says
why, and still carries the cloth mask, which is usually the explanation.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..pipeline import RunOptions, build_pipeline
from ..table import estimate_cloth_color
from ..video import probe, sample_frames
from .workspace import Workspace, build_config


def for_the_app(message: str) -> str:
    """An error written for the command line, reworded for the page."""
    return re.sub(r"(,? or )?[Pp]ass --table-corners to set the four corners by hand",
                  lambda m: (m.group(1) + "place" if m.group(1) else "Place")
                  + " the four corners by hand (Set up, Corners)",
                  message)


def _pts(a: Any) -> List[List[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(a).reshape(-1, 2)]


#: Calibrations by (video, settings), so moving along the video to check
#: another frame does not measure the table all over again.
_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_CACHE_SIZE = 6
_CACHE_LOCK = threading.Lock()
_DETECT_LOCK = threading.Lock()


def _calibrated(path: str, cfg: Any, start: int, end: Optional[int], settings: Dict[str, Any]):
    key = json.dumps([path, settings], sort_keys=True, default=str)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    found = build_pipeline(cfg, RunOptions(
        video=path, start_frame=start, end_frame=end, table_corners=settings["corners"],
    ))
    with _CACHE_LOCK:
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return found


def calibration_report(ws: Workspace, vid: str, settings: Dict[str, Any],
                       t_s: Optional[float] = None) -> Dict[str, Any]:
    video = ws.video(vid)
    if video is None:
        raise KeyError(vid)
    cfg = build_config(settings)
    info = probe(video["path"])
    start = int(round(settings["start_s"] * info.fps))
    end = None if settings["end_s"] is None else int(round(settings["end_s"] * info.fps))
    if t_s is None:
        last = info.frame_count if end is None else min(end, info.frame_count)
        t_s = (start + max(0, last - start) * 0.3) / max(info.fps, 1e-6)
    frame = ws.read_frame(vid, t_s=t_s, max_width=cfg.max_frame_width)
    h, w = frame.shape[:2]
    report: Dict[str, Any] = {
        "ok": False,
        "image_size": [w, h],
        "t_s": round(float(t_s), 3),
        "manual": settings["corners"] is not None,
        "table_size_in": [cfg.table.length_in, cfg.table.width_in],
        "warnings": [],
    }
    try:
        pipeline, calib, _ = _calibrated(video["path"], cfg, start, end, settings)
    except (RuntimeError, ValueError) as exc:
        report["error"] = for_the_app(str(exc))
        return report

    table = calib.table
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    with _DETECT_LOCK:  # the cached detector is shared between requests
        detections = pipeline.detector.detect(frame, hsv)
    centre = tuple(table.corners_image.mean(axis=0))
    ball_r = table.expected_ball_radius_px(centre)
    coverage = pipeline.detector.bed_cloth_coverage(calib.cloth.mask(hsv))

    pockets = []
    if table.has_pockets:
        angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        radius = cfg.detector.pocket_exclusion_ball_diameters * table.ball_diameter_in
        for px, py in table.pockets_table():
            ring = np.column_stack([px + radius * np.cos(angles), py + radius * np.sin(angles)])
            pockets.append(_pts(table.table_to_image(ring)))

    c = calib.cloth
    swatch = cv2.cvtColor(np.uint8([[[int(c.hue) % 180, int(np.clip(c.sat, 0, 255)),
                                      int(np.clip(c.val, 0, 255))]]]), cv2.COLOR_HSV2BGR)[0, 0]
    report.update({
        "ok": True,
        "corners": _pts(table.corners_image),
        "outline": _pts(table.reference_outline),
        "bed": _pts(table.bed_polygon_image(0.0)),
        "pockets": pockets,
        "detections": [
            {"x": round(d.centre_image[0], 1), "y": round(d.centre_image[1], 1),
             "r": round(d.radius_px, 1)}
            for d in detections
        ],
        "cloth": {
            "hue": round(float(c.hue), 1), "sat": round(float(c.sat), 1), "val": round(float(c.val), 1),
            "colour": "#{:02x}{:02x}{:02x}".format(int(swatch[2]), int(swatch[1]), int(swatch[0])),
            "coverage": round(float(coverage), 3),
        },
        "px_per_inch": round(float(table.mean_px_per_inch()), 2),
        "ball_radius_px": round(float(ball_r), 2),
        "frames_used": calib.frames_used,
        "frames_attempted": calib.frames_attempted,
        "corner_spread_px": round(float(calib.corner_spread_px), 1),
        "cushion_noses": calib.cushion_noses,
        "camera": None if table.camera is None else {
            "position_in": [round(float(v), 1) for v in table.camera["position_in"]],
            "focal_px": round(float(table.camera["focal_px"]), 1),
        },
    })
    warn = report["warnings"]
    if not report["manual"] and calib.frames_used < 0.6 * calib.frames_attempted:
        warn.append(f"The table was found in only {calib.frames_used} of {calib.frames_attempted} "
                    "sampled frames. If the camera cuts away a lot, set a start and end time "
                    "around the shots, or place the corners by hand.")
    if not report["manual"] and calib.corner_spread_px > 12:
        warn.append(f"The corners moved {calib.corner_spread_px:.0f} px between sampled frames: "
                    "the camera may pan or zoom. Check the outline on a few frames.")
    if ball_r < 4.0:
        warn.append(f"A ball is only {ball_r:.1f} px across the middle of the table. Raise the "
                    "processing width (or set it to 0 for full size) if the video is larger.")
    if coverage < 0.6:
        warn.append(f"Only {coverage:.0%} of the table looks like cloth on this frame. A player "
                    "may be in the way, or the outline is off the table.")
    if not detections:
        warn.append("No balls were found on this frame. Try another frame, or check the outline.")
    return report


def cloth_mask_jpeg(ws: Workspace, vid: str, settings: Dict[str, Any], t_s: float) -> bytes:
    """The frame with what the tracker takes for cloth tinted, and the rest dimmed."""
    video = ws.video(vid)
    if video is None:
        raise KeyError(vid)
    cfg = build_config(settings)
    frames = sample_frames(video["path"], 9, max_width=cfg.max_frame_width)
    cloth = estimate_cloth_color(frames, cfg)
    frame = ws.read_frame(vid, t_s=t_s, max_width=cfg.max_frame_width)
    mask = cloth.mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) > 0
    out = (frame * 0.35).astype(np.uint8)
    tint = np.array([60, 200, 255], dtype=np.float32)  # BGR amber
    out[mask] = (0.45 * frame[mask] + 0.55 * tint).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""
