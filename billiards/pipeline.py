"""End-to-end tracking pipeline.

Order of operations per frame:

1. detect  -- everything on the bed that is not cloth, split into balls
2. track   -- predict, globally associate, update, spawn, retire
3. events  -- collisions, cushions, pots, shot starts
4. render  -- annotate the camera view and the overhead diagram

Calibration (cloth colour + table homography) happens once up front from frames
sampled across the whole clip, and is only redone if the camera visibly moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .config import Config
from .detect import BallDetector, Detection
from .events import Event, EventDetector, EventType
from .geometry import TableModel
from .render import Renderer
from .table import CalibrationResult, ClothModel, calibrate, largest_cloth_contour
from .track import MultiObjectTracker, Track, TrackState
from .video import (
    TrackCsvWriter,
    VideoInfo,
    VideoSink,
    probe,
    read_frames,
    sample_frames,
    write_json,
)


@dataclass
class FrameResult:
    frame_index: int
    t_s: float
    detections: List[Detection]
    tracks: List[Track]
    events: List[Event]
    annotated: Optional[np.ndarray] = None


class TrackingPipeline:
    """Stateful per-frame tracker.  Own the loop yourself, or use ``run``."""

    def __init__(
        self,
        cfg: Config,
        table: TableModel,
        cloth: ClothModel,
        fps: float,
    ) -> None:
        self.cfg = cfg
        self.table = table
        self.cloth = cloth
        self.fps = max(float(fps), 1.0)
        self.detector = BallDetector(cfg, table, cloth)
        self.tracker = MultiObjectTracker(cfg, table, self.fps)
        self.event_detector = EventDetector(cfg, table)
        self.renderer = Renderer(cfg, table, cloth)
        self.recalibrations = 0
        self._last_frame_index: Optional[int] = None
        self._finished_seen = 0

    # -- per frame ---------------------------------------------------------

    def process(
        self, frame: np.ndarray, frame_index: int, annotate: bool = True
    ) -> FrameResult:
        t_s = frame_index / self.fps
        if self._last_frame_index is None:
            dt = 1.0 / self.fps
        else:
            dt = max(1, frame_index - self._last_frame_index) / self.fps
        self._last_frame_index = frame_index

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if (
            self.cfg.table.recalibration_interval > 0
            and frame_index > 0
            and frame_index % self.cfg.table.recalibration_interval == 0
        ):
            self._maybe_recalibrate(frame, hsv, frame_index)

        detections = self.detector.detect(frame, hsv)
        tracks = self.tracker.update(detections, dt, frame_index, t_s)
        events = self.event_detector.step(tracks, frame_index, t_s)

        # Pots are discovered when the tracker retires a track near a pocket.
        while self._finished_seen < len(self.tracker.finished):
            dead = self.tracker.finished[self._finished_seen]
            self._finished_seen += 1
            if dead.death_reason == "potted":
                events.append(self.event_detector.note_pot(dead, frame_index, t_s))

        # A collision is a discontinuity the motion model cannot represent, so
        # tell the filters to stop trusting their velocity estimates.
        by_id = {t.track_id: t for t in tracks}
        for e in events:
            if e.type is EventType.COLLISION:
                for tid in e.track_ids:
                    if tid in by_id:
                        by_id[tid].kf.apply_impulse()

        annotated = None
        if annotate:
            annotated = self.renderer.draw(
                frame, tracks, detections, events, hud=self.hud(frame_index, t_s)
            )

        return FrameResult(frame_index, t_s, detections, tracks, events, annotated)

    def hud(self, frame_index: int, t_s: float) -> Dict[str, object]:
        stats = self.tracker.last_stats
        return {
            "frame": f"{frame_index}  ({t_s:6.2f}s)",
            "balls": f"{stats.get('confirmed', 0)} tracked, "
                     f"{stats.get('coasting', 0)} coasting",
            "detections": stats.get("detections", 0),
            "events": ", ".join(
                f"{k}={v}" for k, v in sorted(self.event_detector.summary().items())
            )
            or "none",
        }

    # -- recalibration -----------------------------------------------------

    def _maybe_recalibrate(
        self, frame: np.ndarray, hsv: np.ndarray, frame_index: int
    ) -> None:
        """Rebuild the table model if the camera has actually moved.

        A broadcast cuts between angles; a phone on a tripod gets nudged.  Either
        invalidates the homography, and with it every physical threshold.  The
        cheap check below costs one contour fit every few seconds and is what
        keeps a long clip from silently degrading after a camera change.
        """
        from .geometry import quad_from_contour

        mask = self.cloth.mask(hsv)
        contour = largest_cloth_contour(mask)
        if contour is None:
            return
        quad = quad_from_contour(contour)
        if quad is None:
            return

        drift = float(np.max(np.linalg.norm(self.table.corners_image - quad, axis=1)))
        short_side_px = self.cfg.table.width_in * self.table.mean_px_per_inch()
        if drift <= self.cfg.table.recalibration_tolerance * short_side_px:
            return

        self.table = TableModel(
            corners_image=quad,
            length_in=self.cfg.table.length_in,
            width_in=self.cfg.table.width_in,
            ball_diameter_in=self.cfg.table.ball_diameter_in,
            has_pockets=self.table.has_pockets,
        )
        self.detector = BallDetector(self.cfg, self.table, self.cloth)
        self.tracker = MultiObjectTracker(self.cfg, self.table, self.fps)
        self.event_detector = EventDetector(self.cfg, self.table)
        self.renderer = Renderer(self.cfg, self.table, self.cloth)
        self._finished_seen = 0
        self.recalibrations += 1

    # -- summary -----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "table": self.table.to_dict(),
            "cloth": self.cloth.to_dict(),
            "events": self.event_detector.summary(),
            "recalibrations": self.recalibrations,
            "tracks_created": self.tracker._next_id - 1,
            "tracks_alive": len(self.tracker.tracks),
            "tracks_finished": len(self.tracker.finished),
            "finished_reasons": _count(
                [t.death_reason or "unknown" for t in self.tracker.finished]
            ),
        }


def _count(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


# --------------------------------------------------------------------------
# Whole-video driver
# --------------------------------------------------------------------------


@dataclass
class RunOptions:
    video: str
    output: Optional[str] = None
    export_csv: Optional[str] = None
    export_json: Optional[str] = None
    show: bool = False
    debug: bool = False
    start_frame: int = 0
    end_frame: Optional[int] = None
    table_corners: Optional[Sequence[Sequence[float]]] = None
    progress_every: int = 60
    on_progress: Optional[Callable[[str], None]] = None


def build_pipeline(
    cfg: Config, opts: RunOptions
) -> Tuple[TrackingPipeline, CalibrationResult, VideoInfo]:
    info = probe(opts.video)
    frames = sample_frames(
        opts.video,
        cfg.table.calibration_frames,
        start_frame=opts.start_frame,
        end_frame=opts.end_frame,
        max_width=cfg.max_frame_width,
    )

    if opts.table_corners is not None:
        from .table import estimate_cloth_color, table_from_corners

        cloth = estimate_cloth_color(frames, cfg)
        table = table_from_corners(opts.table_corners, cfg)
        result = CalibrationResult(
            table=table,
            cloth=cloth,
            frames_used=len(frames),
            frames_attempted=len(frames),
            corner_spread_px=0.0,
        )
    else:
        result = calibrate(frames, cfg)

    pipeline = TrackingPipeline(cfg, result.table, result.cloth, info.fps)
    return pipeline, result, info


def run(cfg: Config, opts: RunOptions) -> Dict[str, Any]:
    """Process a whole clip, writing whatever outputs were requested."""
    log = opts.on_progress or (lambda msg: None)

    pipeline, calib, info = build_pipeline(cfg, opts)
    centre = tuple(calib.table.corners_image.mean(axis=0))
    log(
        "calibrated from {}/{} frames | cloth HSV {:.0f}/{:.0f}/{:.0f} "
        "| scale {:.1f} px/inch | ball radius {:.1f} px | corner spread {:.1f} px".format(
            calib.frames_used, calib.frames_attempted,
            calib.cloth.hue, calib.cloth.sat, calib.cloth.val,
            calib.table.mean_px_per_inch(),
            calib.table.expected_ball_radius_px(centre),
            calib.corner_spread_px,
        )
    )

    sink = VideoSink(opts.output, info.fps) if opts.output else None
    csv_writer = TrackCsvWriter(opts.export_csv) if opts.export_csv else None
    annotate = bool(opts.output or opts.show)

    window = "Billiard Ball Tracker"
    debug_window = "Debug (foreground mask)"
    paused = False
    frames_done = 0
    t_start = time.time()

    try:
        for idx, frame in read_frames(
            opts.video, opts.start_frame, opts.end_frame, cfg.max_frame_width
        ):
            result = pipeline.process(frame, idx, annotate=annotate)
            frames_done += 1

            if sink is not None and result.annotated is not None:
                sink.write(result.annotated)
            if csv_writer is not None:
                csv_writer.write_frame(idx, result.t_s, result.tracks)

            if opts.show and result.annotated is not None:
                cv2.imshow(window, result.annotated)
                if opts.debug:
                    fg = pipeline.detector.last_debug.get("foreground_mask")
                    if fg is not None:
                        cv2.imshow(debug_window, fg)

                # waitKey(1), not waitKey(0): the original blocked on a key
                # press for every single frame, so the tool could not actually
                # play a video.
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key == ord("q") or key == 27:
                    log("stopped by user")
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("c"):
                    pipeline.tracker.reset_trails()

            if opts.progress_every and frames_done % opts.progress_every == 0:
                elapsed = max(time.time() - t_start, 1e-6)
                log(
                    f"frame {idx}  ({frames_done} processed, "
                    f"{frames_done / elapsed:.1f} fps)  "
                    f"tracks={len(result.tracks)}  "
                    f"events={len(pipeline.event_detector.events)}"
                )
    finally:
        if sink is not None:
            sink.close()
        if csv_writer is not None:
            csv_writer.close()
        if opts.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    summary: Dict[str, Any] = {
        "video": info.to_dict(),
        "calibration": calib.to_dict(),
        "frames_processed": frames_done,
        "wall_seconds": round(elapsed, 2),
        "processing_fps": round(frames_done / elapsed, 2) if elapsed > 0 else None,
        **pipeline.summary(),
        "event_log": [e.to_dict() for e in pipeline.event_detector.events],
    }
    if opts.output:
        summary["output_video"] = str(opts.output)
    if opts.export_csv:
        summary["output_csv"] = str(opts.export_csv)

    if opts.export_json:
        write_json(opts.export_json, summary)
        summary["output_json"] = str(opts.export_json)

    return summary
