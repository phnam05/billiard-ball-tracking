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

from .clock import SourceClock
from .config import Config
from .detect import BallDetector, Detection
from .events import Event, EventDetector, EventType
from .geometry import TableModel
from .render import Renderer
from .shots import Shot, ShotSegmenter
from .table import CalibrationResult, ClothModel, calibrate, corners_on_edge, largest_cloth_contour
from .track import MultiObjectTracker, Track, TrackState
from .video import (
    TrackCsvWriter,
    VideoInfo,
    open_sink,
    probe,
    read_frames,
    sample_frames,
    write_json,
)


def _count_above(diff: np.ndarray, mask: np.ndarray, level: float) -> int:
    """Pixels inside ``mask`` where ``diff`` exceeds ``level`` of full scale.

    Kept in OpenCV rather than NumPy fancy indexing because it runs on every
    frame of every clip: the boolean-mask version of this cost 19 ms a frame,
    which is most of a detection pass.
    """
    threshold = max(1, int(round(level * 255.0)))
    hot = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    return cv2.countNonZero(cv2.bitwise_and(hot, mask))


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
        self.event_detector = EventDetector(cfg, table, self.fps)
        self.renderer = Renderer(cfg, table, cloth)
        self.recalibrations = 0
        self.last_frame_index: Optional[int] = None
        self._finished_seen = 0

        #: Bed cloth coverage when the calibration was fresh, used as the
        #: reference for deciding the view has changed.  Filled on first frame.
        self._reference_coverage: Optional[float] = None
        self._low_coverage_frames = 0
        self._recovery_quads: List[np.ndarray] = []
        #: The previous frame, cropped to the bed, for the two frame-difference
        #: tests below.  Grey is enough to see a cut; the repeat test needs
        #: colour, because a ball can differ from the cloth in hue and not in
        #: luminance.  Both are kept cropped because the bed is well under half
        #: the frame and these run on every frame.
        self._prev_bed: Optional[np.ndarray] = None
        self._prev_bed_grey: Optional[np.ndarray] = None
        self._bed_roi_cache: Optional[Tuple[Any, ...]] = None
        #: False while the calibrated table is not on screen.  Tracking is
        #: suspended rather than producing balls in the crowd.
        self.view_valid = True
        self.view_lost_frames = 0

        #: The last frame that actually carried a measurement, replayed for the
        #: frames that merely repeat it, and how many have been replayed in a
        #: row.  See ``TableConfig.repeat_frame_ball_areas``.
        self._last_measured: Optional[FrameResult] = None
        self._repeat_run = 0
        self.frames_repeated = 0

        #: The scene's clock, which on a screen-recorded broadcast is not the
        #: file's.  A property of the file, so it survives recalibration.
        tcfg = cfg.table
        self.clock = SourceClock(
            self.fps,
            table.ball_radius_in,
            enabled=tcfg.source_clock,
            min_moving_repeats=tcfg.source_clock_min_moving_repeats,
            max_source_frames=tcfg.source_clock_max_skip,
        )
        #: A ball this fast changes the picture on every frame, so a repeat
        #: while one is rolling means the file repeats frames of the scene.
        self._clock_moving_speed = 5.0 * cfg.tracker.stationary_speed_in_s

        #: Run-level totals.  The tracker and event detector are rebuilt from
        #: scratch on a recalibration, so their own counters restart; these
        #: survive so the summary describes the whole clip rather than only the
        #: stretch since the last camera cut.
        self.all_events: List[Event] = []
        self._tracks_created_total = 0
        self.shots = ShotSegmenter(cfg, self.fps)
        self.shots.label_of = self.label_of

    # -- per frame ---------------------------------------------------------

    def process(
        self, frame: np.ndarray, frame_index: int, annotate: bool = True
    ) -> FrameResult:
        t_s = frame_index / self.fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cloth_mask = self.cloth.mask(hsv)
        repainted, repeats = self._bed_change(frame)

        if self._check_view(frame, hsv, cloth_mask, frame_index, repainted):
            # The calibrated table is not on screen.  Reporting tracks here is
            # worse than reporting nothing: on a broadcast cut it draws balls
            # and trajectories over the crowd.
            self.last_frame_index = frame_index
            self._last_measured = None
            self._repeat_run = 0
            annotated = None
            if annotate:
                annotated = self.renderer.compose_idle(
                    frame, self.hud(frame_index, t_s)
                )
            return FrameResult(frame_index, t_s, [], [], [], annotated)

        if (
            repeats
            and self._last_measured is not None
            and self._repeat_run < self.cfg.table.repeat_frame_max_run
        ):
            self._repeat_run += 1
            self.frames_repeated += 1
            self.clock.note_repeat(self.tracker.any_moving(self._clock_moving_speed))
            return self._replay(frame, frame_index, t_s, annotate)
        self._repeat_run = 0

        # Only frames that are measured advance the clock, so the interval the
        # filters integrate over is the interval the picture actually changed
        # across -- however many copies of it the file happened to contain.
        slots = 1 if self.last_frame_index is None else max(1, frame_index - self.last_frame_index)
        self.last_frame_index = frame_index

        if (
            self.cfg.table.recalibration_interval > 0
            and frame_index > 0
            and frame_index % self.cfg.table.recalibration_interval == 0
        ):
            self._maybe_recalibrate(frame, hsv, frame_index)
            cloth_mask = self.cloth.mask(hsv)

        detections = self.detector.detect(frame, hsv, cloth_mask)
        min_step = self.cfg.table.source_clock_min_step_ball_radii * self.table.ball_radius_in
        rate_before = self.clock.source_fps
        dt, _ = self.clock.interval(
            slots,
            moving=self.tracker.any_moving(self._clock_moving_speed),
            fit=lambda intervals: self.tracker.prediction_fit(
                detections, intervals, min_step, spacing_s=1.0 / self.clock.source_fps
            ),
        )
        tracks = self.tracker.update(detections, dt, frame_index, t_s)
        rate_after = self.clock.source_fps
        if rate_after != rate_before:
            # The velocities were learned in the old time base; see clock.py.
            self.tracker.rescale_time(rate_after / rate_before)
        events = self.event_detector.step(tracks, frame_index, t_s, dt)
        self.all_events.extend(events)

        events.extend(self._confirmed_pots())

        # A collision is a discontinuity the motion model cannot represent, so
        # tell the filters to stop trusting their velocity estimates.  Only for
        # a contact found on this frame: one found from the path is a couple of
        # frames old, and the filters have long since followed the ball round.
        by_id = {t.track_id: t for t in tracks}
        for e in events:
            if e.type is EventType.COLLISION and e.frame == frame_index:
                for tid in e.track_ids:
                    if tid in by_id:
                        by_id[tid].kf.apply_impulse()

        self.shots.step(tracks, events, frame_index, t_s)

        annotated = None
        if annotate:
            annotated = self.renderer.draw(
                frame, tracks, detections, events, hud=self.hud(frame_index, t_s)
            )

        result = FrameResult(frame_index, t_s, detections, tracks, events, annotated)
        self._last_measured = result
        return result

    def _confirmed_pots(self) -> List[Event]:
        """Pots, once the tracker has given up waiting for the ball to reappear.

        A track that dies near a pocket waits in the tracker's limbo first
        (``TrackerConfig.revive_window_s``), because a ball in the jaws or
        behind the player's hand dies there too and comes back.  The pot is
        dated to the frame the ball vanished, not the frame it was confirmed.
        """
        pots: List[Event] = []
        while self._finished_seen < len(self.tracker.finished):
            dead = self.tracker.finished[self._finished_seen]
            self._finished_seen += 1
            if dead.death_reason == "potted" and dead.death_frame is not None:
                pot = self.event_detector.note_pot(
                    dead, dead.death_frame, dead.death_frame / self.fps
                )
                pots.append(pot)
                self.all_events.append(pot)
        return pots

    def finish(self) -> List[Event]:
        """End of clip: settle whatever is still waiting, and close the last shot."""
        self.tracker.flush_limbo()
        pots = self._confirmed_pots()
        last = self.last_frame_index or 0
        if pots:
            self.shots.step([], pots, last, last / self.fps)
        self.shots.finish(last, last / self.fps)
        return pots

    def label_of(self, track_id: int) -> str:
        """A track's display label, alive or dead."""
        for track in self.tracker.all_tracks():
            if track.track_id == track_id:
                return track.label
        return f"#{track_id}"

    def _replay(
        self, frame: np.ndarray, frame_index: int, t_s: float, annotate: bool
    ) -> FrameResult:
        """Re-report the last measurement, for a frame that carries no new one.

        The tracks are the same objects, in the same place, so the per-frame
        exports stay one row per ball per frame and the annotated video stays
        one frame per input frame.  The event list is empty rather than a copy:
        nothing happened here, and a caller accumulating ``result.events``
        would otherwise count the same collision two or three times.  The
        drawing still shows the last frame's events, so a contact marker does
        not blink.
        """
        last = self._last_measured
        assert last is not None  # guarded by the caller
        annotated = None
        if annotate:
            annotated = self.renderer.draw(
                frame,
                last.tracks,
                last.detections,
                last.events,
                hud=self.hud(frame_index, t_s),
            )
        return FrameResult(
            frame_index, t_s, last.detections, last.tracks, [], annotated
        )

    def hud(self, frame_index: int, t_s: float) -> Dict[str, object]:
        stats = self.tracker.last_stats
        if not self.view_valid:
            return {
                "frame": f"{frame_index}  ({t_s:6.2f}s)",
                "status": "table not in view - tracking paused",
            }
        return {
            "frame": f"{frame_index}  ({t_s:6.2f}s)",
            "balls": f"{stats.get('confirmed', 0)} tracked, "
                     f"{stats.get('coasting', 0)} coasting",
            "detections": stats.get("detections", 0),
            "events": ", ".join(
                f"{k}={v}" for k, v in sorted(self.event_summary().items())
            )
            or "none",
            "shot": self.shots.live_description() or "none yet",
        }

    # -- view validity -----------------------------------------------------

    def _check_view(
        self,
        frame: np.ndarray,
        hsv: np.ndarray,
        cloth_mask: np.ndarray,
        frame_index: int,
        repainted: bool,
    ) -> bool:
        """Return True when tracking should be suspended for this frame.

        Broadcast pool cuts constantly -- to a replay, to an overhead angle, to
        the players.  Without this check the pipeline keeps applying a stale
        homography, so a cut to a crowd shot produced dozens of "balls" sitting
        on spectators, and the tracker happily drew trajectories between them.
        """
        tcfg = self.cfg.table
        if tcfg.view_change_coverage_ratio <= 0 and not repainted:
            return False

        coverage = self.detector.bed_cloth_coverage(cloth_mask)
        if self._reference_coverage is None:
            self._reference_coverage = max(coverage, 0.2)
            return False

        threshold = tcfg.view_change_coverage_ratio * self._reference_coverage

        if repainted and self.view_valid:
            # A cut needs no patience and no second opinion: nothing that
            # happens *on* a table repaints it.  Acting on the first frame is
            # the whole point -- by the time coverage has collapsed far enough
            # to notice, a dissolve has already been fed to the tracker for a
            # dozen frames, which is long enough to confirm phantom tracks.
            self.view_valid = False
            self._low_coverage_frames = 0
            self._recovery_quads.clear()
        elif coverage >= threshold and self.view_valid:
            self._low_coverage_frames = 0
            self._recovery_quads.clear()
            return False

        if self.view_valid:
            self._low_coverage_frames += 1
            if self._low_coverage_frames < max(1, tcfg.view_change_patience):
                # A player leaning across the table dips coverage for a frame or
                # two; that is not a cut, and tracks should coast through it.
                return False
            self.view_valid = False
            self._recovery_quads.clear()

        self.view_lost_frames += 1
        if repainted:
            # Mid-transition.  The quad fitted from a frame that is half one
            # shot and half another describes neither, so recovery does not
            # even start until the picture settles.
            self._recovery_quads.clear()
            return True

        # Look for the table in the new view.  Adopting it takes several
        # agreeing frames, exactly as the initial calibration does -- a single
        # frame during a crossfade is a bad basis for a homography that
        # everything downstream depends on.
        if self._try_recover(frame, hsv, frame_index):
            self.view_valid = True
            self._low_coverage_frames = 0
            return False
        return True

    def _bed_change(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """How the bed differs from the previous frame, at both ends of the scale.

        Cloth coverage answers "does the bed still look like cloth?", which a
        dissolve between two shots of the same table passes for most of its
        length -- the incoming angle is mostly cloth too.  One difference image
        answers two questions coverage cannot, so it is computed once here.

        **Was the bed repainted?**  Then the camera cut.  The test works
        because of how little of a table a game actually moves: a ball is a
        thousandth of the bed and a player leaning over it is a few percent;
        measured across three clips, the busiest frame of play repaints 6% of
        the bed, against 16-32% for a cut, because every pixel is being mixed
        with a different scene at once.

        **Did the bed change at all?**  Then this frame is a copy of the one
        before it and holds no new measurement -- see
        ``TableConfig.repeat_frame_ball_areas``.  This one is measured across
        the colour channels rather than on luminance, because a ball can differ
        from the cloth in hue and barely at all in brightness, and that ball
        moving is exactly what must not be mistaken for nothing happening.

        A cut clears the repeat test by four orders of magnitude, so the two
        can never both fire.
        """
        tcfg = self.cfg.table
        bed, bed_area = self._bed_roi(frame.shape)
        if bed_area == 0:
            self._prev_bed = self._prev_bed_grey = None
            return False, False

        x, y, w, h = self._bed_roi_cache[1]
        # Copied, not referenced: a caller decoding into a reused buffer would
        # otherwise hand us the same array twice, every frame would compare
        # equal to itself, and the pipeline would replay the first frame for
        # the whole clip.
        patch = frame[y : y + h, x : x + w].copy()
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        prev, self._prev_bed = self._prev_bed, patch
        prev_grey, self._prev_bed_grey = self._prev_bed_grey, grey
        if prev is None or prev.shape != patch.shape:
            return False, False

        repainted = False
        if tcfg.view_change_area_ratio > 0:
            changed = _count_above(cv2.absdiff(grey, prev_grey), bed,
                                   tcfg.view_change_level)
            repainted = changed / float(bed_area) >= tcfg.view_change_area_ratio

        repeats = False
        if tcfg.repeat_frame_max_run > 0:
            # A ball area, from the homography, is the unit here for the same
            # reason it is everywhere else: it makes the threshold mean the
            # same thing at any resolution or camera distance.
            r_px = self.table.expected_ball_radius_px(
                tuple(self.table.corners_image.mean(axis=0))
            )
            budget = tcfg.repeat_frame_ball_areas * np.pi * r_px * r_px
            step = cv2.absdiff(patch, prev)
            # The largest step over the channels, not their average: a ball
            # can move a lot in one channel and nothing in the mean.
            step = cv2.max(cv2.max(step[:, :, 0], step[:, :, 1]), step[:, :, 2])
            repeats = _count_above(step, bed, tcfg.repeat_frame_level) <= budget
        return repainted, repeats

    def _bed_roi(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
        """The bed mask cropped to its own bounding box, and its area.

        Cached on the mask itself, which the detector already caches and
        replaces whenever the geometry changes.
        """
        mask = self.detector.bed_mask(shape)
        if self._bed_roi_cache is None or self._bed_roi_cache[0] is not mask:
            bx, by, bw, bh = cv2.boundingRect(mask)
            self._bed_roi_cache = (
                mask,
                (bx, by, bw, bh),
                mask[by : by + bh, bx : bx + bw],
                int(cv2.countNonZero(mask)),
            )
        return self._bed_roi_cache[2], self._bed_roi_cache[3]

    def _bed_repainted(self, frame: np.ndarray) -> bool:
        """Has most of the bed changed since the previous frame?  (A cut has.)"""
        return self._bed_change(frame)[0]

    # -- recalibration -----------------------------------------------------

    def _fit_quad(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        from .geometry import quad_from_contour

        contour = largest_cloth_contour(self.cloth.mask(hsv))
        if contour is None:
            return None
        return quad_from_contour(contour)

    def _adopt_table(self, quad: np.ndarray, frame: np.ndarray, hsv: np.ndarray) -> bool:
        """Validate a candidate table polygon, and switch to it if it holds up.

        A cloth-coloured blob in a crowd shot will happily yield *some*
        quadrilateral.  Requiring the quad to be almost entirely cloth is what
        separates a table from a sponsor banner, and stops a bad recalibration
        from replacing a good calibration with nonsense.
        """
        unchanged = self._same_table(quad)
        if unchanged:
            candidate, candidate_detector = self.table, self.detector
        else:
            # After a cut to a low close-up the cloth runs off the picture;
            # an outline fitted to that took in the score bar, whose ball
            # icons were then tracked (2025 UK Open, 28 Sep 2026).
            if corners_on_edge(quad, frame.shape[1], frame.shape[0]) >= 2:
                return False
            candidate = TableModel(
                corners_image=quad,
                length_in=self.cfg.table.length_in,
                width_in=self.cfg.table.width_in,
                ball_diameter_in=self.cfg.table.ball_diameter_in,
                has_pockets=self.table.has_pockets,
                image_size=(frame.shape[1], frame.shape[0]),
                ball_parallax=self.cfg.table.ball_parallax,
            )
            if candidate.expected_ball_radius_px(tuple(quad.mean(axis=0))) < self.cfg.table.min_ball_radius_px:
                return False
            candidate_detector = BallDetector(self.cfg, candidate, self.cloth)

        coverage = candidate_detector.bed_cloth_coverage(self.cloth.mask(hsv))
        if coverage < self.cfg.table.min_bed_coverage:
            return False

        if unchanged:
            # The view went away and came back on the same table -- a replay,
            # a dissolve that resolved to the shot it started from, a hand
            # over the lens.  Rebuilding here would throw away every ball's
            # identity for nothing, so tracking simply resumes.
            self._reference_coverage = None
            self._recovery_quads.clear()
            return True

        # Settle anything waiting in the old tracker's limbo before it goes.
        self.tracker.flush_limbo()
        late = self._confirmed_pots()
        if late:
            last = self.last_frame_index or 0
            self.shots.step([], late, last, last / self.fps)
        self._tracks_created_total += self.tracker.tracks_created
        self.table = candidate
        self.detector = candidate_detector
        self.tracker = MultiObjectTracker(self.cfg, self.table, self.fps)
        self.event_detector = EventDetector(self.cfg, self.table, self.fps)
        self.renderer = Renderer(self.cfg, self.table, self.cloth)
        self._finished_seen = 0
        self.recalibrations += 1
        self._reference_coverage = None
        self._recovery_quads.clear()
        return True

    def _same_table(self, quad: np.ndarray) -> bool:
        """Is this candidate the table we are already calibrated to?"""
        drift = float(np.max(np.linalg.norm(self.table.reference_outline - quad, axis=1)))
        short_side_px = self.cfg.table.width_in * self.table.mean_px_per_inch()
        return drift <= self.cfg.table.recalibration_tolerance * short_side_px

    def _try_recover(
        self, frame: np.ndarray, hsv: np.ndarray, frame_index: int
    ) -> bool:
        """Re-find the table after a cut, from several agreeing frames."""
        tcfg = self.cfg.table
        quad = self._fit_quad(hsv)
        if quad is None:
            return False

        self._recovery_quads.append(quad)
        need = max(1, tcfg.recovery_frames)
        if len(self._recovery_quads) < need:
            return False

        recent = np.stack(self._recovery_quads[-need:], axis=0)
        corners = np.median(recent, axis=0)
        spread = float(np.max(np.linalg.norm(recent - corners, axis=2)))
        # Still settling (a crossfade, a pan): drop the oldest and keep looking.
        if spread > tcfg.recovery_max_spread_px:
            self._recovery_quads.pop(0)
            return False

        return self._adopt_table(corners, frame, hsv)

    def _maybe_recalibrate(
        self, frame: np.ndarray, hsv: np.ndarray, frame_index: int
    ) -> bool:
        """Periodic drift check while the table *is* in view.

        A tripod gets nudged, a broadcast camera slowly zooms.  Either
        invalidates the homography and with it every physical threshold, so the
        geometry is re-checked every few seconds and only rebuilt if it has
        actually moved.
        """
        quad = self._fit_quad(hsv)
        if quad is None:
            return False

        if self._same_table(quad):
            return False
        if corners_on_edge(quad, frame.shape[1], frame.shape[0]) >= 2:
            # The camera has zoomed in until the cloth runs off the picture.
            # There is no table to adopt, and the old one is not where the
            # table is any more: tracked on it, the 2025 UK Open's push-in
            # put a row of "balls" on a rail and 95 contacts between them.
            # So tracking waits, as after a cut, for the whole table again.
            self.view_valid = False
            self._low_coverage_frames = 0
            self._recovery_quads.clear()
            return False
        return self._adopt_table(quad, frame, hsv)

    # -- summary -----------------------------------------------------------

    def event_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.all_events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        return {
            "table": self.table.to_dict(),
            "cloth": self.cloth.to_dict(),
            "events": self.event_summary(),
            "recalibrations": self.recalibrations,
            "frames_view_lost": self.view_lost_frames,
            "frames_repeated": self.frames_repeated,
            "clock": self.clock.to_dict(),
            "shots": len(self.shots.shots),
            "tracks_created": self._tracks_created_total + self.tracker.tracks_created,
            "tracks_alive": len(self.tracker.tracks),
            "tracks_finished": len(self.tracker.finished),
            "tracks_revived": self.tracker.revived,
            "ball_set": self.tracker.ball_set,
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
    #: How the annotated video is encoded: a fourcc, ``"ffmpeg"`` (H.264 a
    #: browser plays, see ``video.browser_codec``), or None for the suffix's.
    writer: Optional[str] = None
    #: Called after every frame with its result and the pipeline -- the app's
    #: progress bar and live preview.  ``annotate`` draws the frames for it
    #: even when no video is written.
    on_frame: Optional[Callable[["FrameResult", "TrackingPipeline"], None]] = None
    annotate: bool = False
    #: Polled after every frame; True ends the run early, keeping what was
    #: measured so far.
    should_stop: Optional[Callable[[], bool]] = None


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

        h, w = frames[0].shape[:2]
        table = table_from_corners(opts.table_corners, cfg, image_size=(w, h))
        # The corners say where the bed is, so the cloth is measured there: in
        # a wide shot the most common saturated colour in the frame can be the
        # floor or a banner, which is often why the table was not found.
        bed = table.bed_mask((h, w), margin_in=2.0 * table.ball_diameter_in)
        inside = [cv2.bitwise_and(f, f, mask=bed) for f in frames]
        try:
            cloth = estimate_cloth_color(inside, cfg, neutral_ok=True)
        except RuntimeError:
            cloth = estimate_cloth_color(frames, cfg)
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

    sink = open_sink(opts.output, info.fps, opts.writer) if opts.output else None
    csv_writer = TrackCsvWriter(opts.export_csv) if opts.export_csv else None
    annotate = bool(opts.output or opts.show or opts.annotate)
    stopped = False

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
            if opts.on_frame is not None:
                opts.on_frame(result, pipeline)
            if opts.should_stop is not None and opts.should_stop():
                log("stopped early")
                stopped = True
                break

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
                    f"events={len(pipeline.all_events)}"
                )
    finally:
        pipeline.finish()
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
        "stopped_early": stopped,
        "wall_seconds": round(elapsed, 2),
        "processing_fps": round(frames_done / elapsed, 2) if elapsed > 0 else None,
        **pipeline.summary(),
        "shot_log": pipeline.shots.to_list(),
        "event_log": [e.to_dict() for e in pipeline.all_events],
    }
    if opts.output:
        summary["output_video"] = str(opts.output)
    if opts.export_csv:
        summary["output_csv"] = str(opts.export_csv)

    if opts.export_json:
        write_json(opts.export_json, summary)
        summary["output_json"] = str(opts.export_json)

    return summary
