"""Multi-object tracking with identity preservation.

This module is where most of the robustness comes from.  The previous pipeline
had no tracker: each frame it recomputed "the largest blob" and appended the
result to a list if it happened to be within 100 pixels of the last one.  That
means a single missed detection permanently broke the trajectory, two balls
crossing swapped identities, and anything occluded was simply gone.

Here every ball gets a Kalman filter in table space, detections are assigned to
tracks by globally optimal matching on distance *and* colour, and a track that
loses its detection coasts on its motion model instead of dying.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import balls as ballnum
from .assignment import FORBIDDEN, associate
from .config import Config
from .detect import ColorSignature, Detection, colour_distance_matrix
from .geometry import TableModel
from .kalman import BallKalman


#: A ball's own colour has to be at least this light and at most this
#: colourful before it is called the cue ball, so a table with no cue ball in
#: frame does not promote the palest solid.  Measured: cue balls 148-225 and
#: chroma 5-13; the palest solid, the yellow 1, has chroma 46-48.
_MIN_CUE_LIGHTNESS = 140.0
_MAX_CUE_CHROMA = 20.0

#: ...but the ball already holding a role only loses it below this fraction of
#: the bar it had to clear.  A cue ball resting beside a pocket samples the
#: pocket's shadow round its edge; on albin_fedor its white fraction sank from
#: 0.77 to 0.41 there, and it stopped being the cue ball for the rest of the
#: clip.  Being the cue ball should not flicker with a shadow.
_ROLE_HOLD_RATIO = 0.6

#: Fraction of its speed normal to the rail a ball keeps through a cushion.
_CUSHION_RESTITUTION = 0.8

#: ...and at least this much of it black, and at most this colourful, before
#: anything is called the 8.  Measured: the 8 is 0.58-0.89 black with chroma
#: 1-6; the darkest solid, the blue 2, 0.13 black with chroma 19-28.
_MIN_EIGHT_BLACK = 0.4
_MAX_EIGHT_CHROMA = 14.0

#: The evidence for each ball set is remembered with this weight per frame, and
#: the set in use changes only once another is ahead by this much (in squared
#: cost units: one ball fitting 2 units better for one frame is 4).
_SET_EVIDENCE_DECAY = 0.98
_SET_SWITCH_MARGIN = 6.0

#: How much better a rival has to score before a role changes hands.  Two
#: similar-looking balls otherwise trade the "CUE" label back and forth every
#: few frames, which is worse than being slightly wrong consistently.
_ROLE_STICKINESS = 0.10


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class TrackSample:
    frame: int
    t_s: float
    table_xy: Tuple[float, float]
    image_xy: Tuple[float, float]
    speed_in_s: float
    observed: bool


class Track:
    """One ball, over time."""

    __slots__ = (
        "track_id", "kf", "signature", "state", "hits", "age",
        "time_since_update", "trail", "last_image_xy", "last_radius_px",
        "birth_frame", "death_frame", "death_reason", "_trail_cap",
        "last_observed_xy", "role", "number", "clean_samples", "band_hits",
    )

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        cfg: Config,
        frame: int,
        trail_cap: int,
    ) -> None:
        self.track_id = track_id
        tc = cfg.tracker
        self.kf = BallKalman(
            detection.centre_table,
            velocity_tau_s=tc.velocity_tau_s,
            accel_std_in_s2=tc.accel_std_in_s2,
            meas_std_in=tc.meas_std_in,
            init_vel_std_in_s=tc.init_vel_std_in_s,
            manoeuvre_gain_max=tc.manoeuvre_gain_max,
        )
        self.signature = detection.signature
        #: Detections of this ball on its own, not split out of a cluster.
        #: Only those are trusted with its colour: see ``update``.
        self.clean_samples = 1 if detection.colour_ok else 0
        self.state = TrackState.TENTATIVE
        self.hits = 1
        #: Detections past the bed's far edge (``Detection.in_raised_band``).
        self.band_hits = 1 if detection.in_raised_band else 0
        self.age = 1
        self.time_since_update = 0
        self._trail_cap = trail_cap
        self.trail: Deque[TrackSample] = deque(maxlen=trail_cap)
        self.last_image_xy = detection.centre_image
        self.last_observed_xy = detection.centre_table
        self.last_radius_px = detection.radius_px
        self.birth_frame = frame
        self.death_frame: Optional[int] = None
        self.death_reason: Optional[str] = None
        #: 'cue', 'eight' or None -- assigned across all tracks at once,
        #: because a table has exactly one of each.
        self.role: Optional[str] = None
        #: The number printed on the ball, when its colour says which one it
        #: is -- see ``MultiObjectTracker.assign_numbers``.
        self.number: Optional[int] = None

    # -- properties --------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed

    @property
    def is_alive(self) -> bool:
        return self.state is not TrackState.DELETED

    @property
    def is_visible(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    @property
    def ball_type(self) -> str:
        """``cue``, ``eight``, ``stripe`` or ``solid``.

        The first two come from the table-wide role assignment rather than from
        this ball's appearance alone, so at most one track can ever be the cue
        ball and at most one the 8.
        """
        if self.role:
            return self.role
        if self.number is not None:
            return "stripe" if self.number > 8 else "solid"
        return "stripe" if self.signature.classify() == "stripe" else "solid"

    @property
    def label(self) -> str:
        """``CUE``, ``8``, the ball's number, or ``#<track id>`` if unknown."""
        if self.role == "cue":
            return "CUE"
        if self.role == "eight":
            return "8"
        if self.number is not None:
            return str(self.number)
        return f"#{self.track_id}"

    # -- filter interaction ------------------------------------------------

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, cfg: Config) -> None:
        self.kf.update(detection.centre_table)
        self.hits += 1
        if detection.in_raised_band:
            self.band_hits += 1
        self.time_since_update = 0
        self.last_image_xy = detection.centre_image
        self.last_observed_xy = detection.centre_table
        self.last_radius_px = detection.radius_px

        # Colour is only learned from detections of this ball on its own.  One
        # split out of a cluster samples its neighbours round its rim, so in a
        # rack every ball reads as a stripe of some mixed colour; a track born
        # there used to keep that colour and fade it out at 8% a frame, and was
        # still being named from it a dozen frames after the break.  So the
        # first clean sample replaces whatever came before, the next few are
        # averaged in evenly, and after that the average moves slowly: a
        # detection clipped by the cue stick or taken mid-collision has a
        # contaminated colour, and a fast average would let it poison the
        # identity the tracker relies on.
        if detection.colour_ok:
            self.clean_samples += 1
            if self.clean_samples == 1:
                self.signature = detection.signature
            else:
                alpha = max(0.08, 1.0 / self.clean_samples)
                self.signature = self.signature.blend(detection.signature, alpha)

        if self.state is TrackState.TENTATIVE:
            # Seen mostly past the far edge, it never rolled there from the
            # bed: a hand on the rail, or a ball's own top in the shadow under
            # the cushion's nose.  It is never confirmed.
            if self.hits >= cfg.tracker.min_hits_to_confirm and 2 * self.band_hits <= self.hits:
                self.state = TrackState.CONFIRMED
        elif self.state is TrackState.COASTING:
            self.state = TrackState.CONFIRMED

    def mark_missed(self, cfg: Config) -> None:
        if self.state is TrackState.CONFIRMED:
            self.state = TrackState.COASTING

    def kill(self, frame: int, reason: str) -> None:
        self.state = TrackState.DELETED
        self.death_frame = frame
        self.death_reason = reason

    def record(self, frame: int, t_s: float, table: TableModel, observed: bool) -> None:
        pos = self.kf.position
        img = table.ball_table_to_image([tuple(pos)])[0]
        self.last_image_xy = (float(img[0]), float(img[1]))
        self.trail.append(
            TrackSample(
                frame=frame,
                t_s=t_s,
                table_xy=(float(pos[0]), float(pos[1])),
                image_xy=self.last_image_xy,
                speed_in_s=self.speed,
                observed=observed,
            )
        )

    def to_dict(self) -> dict:
        pos = self.kf.position
        vel = self.kf.velocity
        return {
            "id": self.track_id,
            "state": self.state.value,
            "type": self.ball_type,
            "label": self.label,
            "number": self.number,
            "x_in": round(float(pos[0]), 3),
            "y_in": round(float(pos[1]), 3),
            "vx_in_s": round(float(vel[0]), 2),
            "vy_in_s": round(float(vel[1]), 2),
            "speed_in_s": round(self.speed, 2),
            "hits": self.hits,
            "age": self.age,
            "colour": self.signature.to_dict(),
        }


class MultiObjectTracker:
    def __init__(self, cfg: Config, table: TableModel, fps: float) -> None:
        self.cfg = cfg
        self.table = table
        self.fps = max(fps, 1.0)
        self._next_id = 1
        self.tracks: List[Track] = []
        self.finished: List[Track] = []
        #: Tracks that just died, kept for a moment in case their ball comes
        #: back.  See ``_revive``.
        self.limbo: List[Track] = []
        self.revived = 0
        self._limbo_frames = max(1, int(round(cfg.tracker.revive_window_s * self.fps)))
        trail_seconds = cfg.render.trail_seconds if cfg.render.trail_seconds > 0 else 8.0
        self._trail_cap = int(max(8, round(trail_seconds * self.fps)))
        self.last_stats: Dict[str, int] = {}
        self._role_holders: Dict[str, int] = {}
        bc = cfg.balls
        #: Which ball set the table is played with, and how well each fits.
        self._ball_sets = list(ballnum.BALL_SETS) if bc.ball_set == "auto" else [bc.ball_set]
        self.ball_set: Optional[str] = None if bc.ball_set == "auto" else bc.ball_set
        self._set_fit: Dict[str, float] = {}
        self._set_scales: Dict[str, Tuple[float, float]] = {s: (1.0, 1.0) for s in self._ball_sets}
        self._allowed_numbers = ballnum.parse_numbers(bc.numbers)

    # -- cost --------------------------------------------------------------

    def _gate_inches(self, dt: float) -> float:
        tc = self.cfg.tracker
        return (
            tc.max_speed_in_s * dt
            + tc.gate_padding_ball_diameters * self.table.ball_diameter_in
        )

    def _cost_matrix(self, detections: Sequence[Detection], dt: float) -> np.ndarray:
        tc = self.cfg.tracker
        gate = self._gate_inches(dt)
        n, m = len(self.tracks), len(detections)
        cost = np.full((n, m), FORBIDDEN, dtype=np.float64)
        if n == 0 or m == 0:
            # np.array([]) has shape (0,), not (0, 2), so the broadcast below
            # would raise.  A frame with no detections is completely normal --
            # an empty table, a heavy occlusion, a paused view.
            return cost

        det_xy = np.array(
            [d.centre_table for d in detections], dtype=np.float64
        ).reshape(m, 2)
        pred_xy = np.array(
            [t.kf.position for t in self.tracks], dtype=np.float64
        ).reshape(n, 2)

        dist = np.linalg.norm(pred_xy[:, None, :] - det_xy[None, :, :], axis=2)
        colour = colour_distance_matrix(
            [t.signature for t in self.tracks],
            [d.signature for d in detections],
        )

        # A pair is only a candidate if the ball could physically have moved
        # that far in one frame *and* it still looks like the same ball.
        allowed = (dist <= gate) & (colour <= tc.max_color_distance)
        scored = dist / gate + tc.color_cost_weight * (colour / tc.max_color_distance)
        return np.where(allowed, scored, cost)

    # -- main step ---------------------------------------------------------

    def update(
        self,
        detections: Sequence[Detection],
        dt: float,
        frame: int,
        t_s: float,
    ) -> List[Track]:
        dt = float(max(dt, 1e-4))
        for track in self.tracks:
            track.predict(dt)

        cost = self._cost_matrix(detections, dt)
        matches, unmatched_tracks, unmatched_dets = associate(cost)

        matched_track_idx = set()
        for ti, di in matches:
            self.tracks[ti].update(detections[di], self.cfg)
            matched_track_idx.add(ti)

        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed(self.cfg)
            # Only a ball that was not seen: a seen one is where it was seen.
            self._bounce_off_rails(self.tracks[ti])

        for di in unmatched_dets:
            if self._revive(detections[di], frame):
                continue
            # Past the far edge only balls already being followed are looked
            # for: a hand on the far rail is there too, and is never a ball
            # that rolled there.
            if not detections[di].in_raised_band:
                self._spawn(detections[di], frame)

        self._retire(frame)
        self._expire_limbo(frame)
        self.assign_roles()
        self.assign_numbers()

        for track in self.tracks:
            track.record(frame, t_s, self.table, observed=track.time_since_update == 0)

        self.last_stats = {
            "detections": len(detections),
            "matched": len(matches),
            "new": len(unmatched_dets),
            "coasting": sum(1 for t in self.tracks if t.state is TrackState.COASTING),
            "confirmed": sum(1 for t in self.tracks if t.state is TrackState.CONFIRMED),
            "tentative": sum(1 for t in self.tracks if t.state is TrackState.TENTATIVE),
        }
        return self.active_tracks()

    def _bounce_off_rails(self, track: Track) -> None:
        """Reflect an unseen ball's prediction off any cushion it has reached.

        A ball is least likely to be seen exactly while it bounces off the far
        rail (see ``geometry.TableModel.bed_mask``).  Predicted straight on, it
        sails through the cushion, and when it reappears after bouncing it is
        nowhere near its prediction and becomes a new ball -- or, next to a
        side pocket, it is predicted off the table and reported potted.  Only
        the mean is reflected; the filter's uncertainty is left as it is.
        """
        r = self.table.ball_radius_in
        x = track.kf.x
        inside = (r <= x[0] <= self.table.length_in - r) and (r <= x[1] <= self.table.width_in - r)
        if inside:
            return
        # A ball heading into a pocket is not bouncing: reflecting it back onto
        # the bed would hide the pot.  "Heading into" rather than "near",
        # because a ball entering a corner pocket along the rail crosses the
        # rail line several inches before it reaches the pocket.
        if self._heading_into_pocket(x[:2], x[2:]):
            return
        for axis, limit in ((0, self.table.length_in), (1, self.table.width_in)):
            if x[axis] < r:
                x[axis] = 2.0 * r - x[axis]
                x[axis + 2] = abs(x[axis + 2]) * _CUSHION_RESTITUTION
            elif x[axis] > limit - r:
                x[axis] = 2.0 * (limit - r) - x[axis]
                x[axis + 2] = -abs(x[axis + 2]) * _CUSHION_RESTITUTION

    def _heading_into_pocket(self, pos: np.ndarray, vel: np.ndarray, horizon_s: float = 0.5) -> bool:
        if not self.table.has_pockets:
            return False
        reach = self.cfg.events.pocket_radius_ball_diameters * self.table.ball_diameter_in
        speed_sq = float(np.dot(vel, vel))
        for pocket in self.table.pockets_table():
            gap = pocket - pos
            s = 0.0 if speed_sq < 1e-9 else float(np.clip(np.dot(gap, vel) / speed_sq, 0.0, horizon_s))
            if float(np.linalg.norm(gap - vel * s)) <= reach:
                return True
        return False

    def _spawn(self, detection: Detection, frame: int) -> None:
        track = Track(self._next_id, detection, self.cfg, frame, self._trail_cap)
        self._next_id += 1
        self.tracks.append(track)

    def _retire(self, frame: int) -> None:
        tc = self.cfg.tracker
        ecfg = self.cfg.events
        alive: List[Track] = []
        for track in self.tracks:
            reason: Optional[str] = None

            if track.state is TrackState.TENTATIVE:
                if track.time_since_update > tc.max_age_tentative:
                    reason = "spurious"
            elif track.time_since_update > self._coasting_budget(track):
                reason = "lost"

            # A coasting track predicted well off the bed is either potted or a
            # bad extrapolation; either way it should not keep drawing a line.
            pos = track.kf.position
            if reason is None and track.time_since_update > 0:
                margin = -1.5 * self.table.ball_radius_in
                if not self.table.contains((float(pos[0]), float(pos[1])), margin):
                    reason = "off_table"

            if reason == "lost" or reason == "off_table":
                pocket_d = self.table.nearest_pocket_distance(
                    (float(pos[0]), float(pos[1]))
                )
                if pocket_d <= (
                    ecfg.pocket_radius_ball_diameters * self.table.ball_diameter_in
                ):
                    reason = "potted"

            if reason is not None:
                track.kill(frame, reason)
                # A blob that was never confirmed is not worth waiting for.
                (self.finished if reason == "spurious" else self.limbo).append(track)
            else:
                alive.append(track)
        self.tracks = alive

    def _revive(self, detection: Detection, frame: int) -> bool:
        """Give a reappearing ball back the identity it had.

        A ball that stops in the pocket jaws is inside the region detection
        ignores, so its track dies there -- and was reported potted -- and when
        it rolls back out it became a brand-new ball.  On albin_fedor that
        turned the cue ball into "#10" for the rest of the clip and logged a
        scratch that never happened.  The same happens to a ball hidden by the
        player for longer than its coasting budget.  So a track that dies waits
        here for ``revive_window_s``; a new detection of the same colour close
        to where it vanished is that ball, not a new one, and its death (and
        any pot) is withdrawn.
        """
        if not self.limbo:
            return False
        tc = self.cfg.tracker
        det = np.asarray(detection.centre_table, dtype=np.float64)
        colours = colour_distance_matrix([t.signature for t in self.limbo], [detection.signature])[:, 0]
        best, best_cost = None, None
        for track, colour in zip(self.limbo, colours):
            # A ball that vanished while rolling -- off the far rail, where it
            # cannot be seen -- turns up further along, so the reach grows with
            # how fast it was going and how long it has been gone.
            gone_s = max(0, frame - (track.death_frame or frame)) / self.fps
            reach = (
                tc.revive_distance_ball_diameters * self.table.ball_diameter_in
                + track.speed * gone_s
            )
            gap = float(np.linalg.norm(np.asarray(track.last_observed_xy) - det))
            if gap > reach or colour > tc.max_color_distance:
                continue
            cost = gap / reach + colour / tc.max_color_distance
            if best_cost is None or cost < best_cost:
                best, best_cost = track, cost
        if best is None:
            return False
        self.limbo.remove(best)
        best.kf = BallKalman(
            detection.centre_table,
            velocity_tau_s=tc.velocity_tau_s,
            accel_std_in_s2=tc.accel_std_in_s2,
            meas_std_in=tc.meas_std_in,
            init_vel_std_in_s=tc.init_vel_std_in_s,
            manoeuvre_gain_max=tc.manoeuvre_gain_max,
        )
        best.state = TrackState.CONFIRMED
        best.death_frame = None
        best.death_reason = None
        best.update(detection, self.cfg)
        self.tracks.append(best)
        self.revived += 1
        return True

    def _expire_limbo(self, frame: int) -> None:
        keep: List[Track] = []
        for track in self.limbo:
            if track.death_frame is not None and frame - track.death_frame >= self._limbo_frames:
                self.finished.append(track)
            else:
                keep.append(track)
        self.limbo = keep

    def flush_limbo(self) -> None:
        """End of clip: nothing more can come back."""
        self.finished.extend(self.limbo)
        self.limbo = []

    def _coasting_budget(self, track: Track) -> float:
        """How many frames this track may go unseen before it is dropped.

        A coasting track is being extrapolated from its motion model, and how
        far that is worth trusting depends on how much was measured to build
        it.  A ball watched for five hundred frames has earned the full window
        -- it needs it, because the player's body hides it for most of a
        stroke.  A blob that was confirmed on three frames has earned three.
        """
        tc = self.cfg.tracker
        return min(float(tc.max_age_coasting), tc.coast_frames_per_hit * track.hits)

    # -- clock evidence ----------------------------------------------------

    def rescale_time(self, factor: float) -> None:
        """Every ball's velocity, in a clock ``factor`` times faster."""
        for track in list(self.tracks) + list(self.limbo):
            track.kf.rescale_time(factor)

    def _clock_tracks(self, min_speed_in_s: float, settled: bool = False) -> List[Track]:
        """Balls that are confirmed, were seen last frame, and are moving --
        and, with ``settled``, whose motion model saw that frame coming."""
        return [
            t for t in self.tracks
            if t.state is TrackState.CONFIRMED
            and t.time_since_update == 0
            and t.speed >= min_speed_in_s
            and (not settled or t.kf.settled)
        ]

    def any_moving(self, min_speed_in_s: float) -> bool:
        return bool(self._clock_tracks(min_speed_in_s))

    def prediction_fit(
        self,
        detections: Sequence[Detection],
        intervals_s: Sequence[float],
        min_step_in: float,
        spacing_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """How well each candidate interval explains where the moving balls went.

        For every ball moving fast enough that one candidate interval puts it
        at least ``min_step_in`` further on than the next, predict it forward
        by each interval and measure the distance to the nearest detection.
        Returns the median of those distances per interval, in inches, or None
        if nothing is moving fast enough to tell the intervals apart.  Used by
        :class:`billiards.clock.SourceClock` -- a moving ball is a clock.
        """
        if not detections or not intervals_s:
            return None
        # The intervals that matter are ``spacing_s`` apart (a source frame),
        # however finely they are sampled.
        spacing = spacing_s if spacing_s else min(
            (b - a for a, b in zip(intervals_s, intervals_s[1:])),
            default=intervals_s[0],
        )
        # Only balls whose motion model is settled.  One just struck is
        # accelerating while its filter still holds the old velocity, so it
        # lands further on than one source frame predicts, and "two frames"
        # wins decisively: on a four-second clip whose decisive frames came
        # mostly from the break, that read 47 fps off 25 fps content.
        movers = self._clock_tracks(min_step_in / max(spacing, 1e-6), settled=True)
        if not movers:
            return None
        det = np.array([d.centre_table for d in detections], dtype=np.float64).reshape(-1, 2)
        cap = 2.0 * self.table.ball_diameter_in
        errors = np.empty((len(movers), len(intervals_s)), dtype=np.float64)
        for i, track in enumerate(movers):
            pred = track.kf.peek_many(intervals_s)  # (intervals, 2)
            gaps = np.linalg.norm(pred[:, None, :] - det[None, :, :], axis=2)
            errors[i] = np.minimum(np.min(gaps, axis=1), cap)
        return np.median(errors, axis=0)

    # -- accessors ---------------------------------------------------------

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.is_visible]

    def all_tracks(self) -> List[Track]:
        return list(self.tracks) + list(self.limbo) + list(self.finished)

    def assign_roles(self) -> None:
        """Decide which single track is the cue ball, and which is the 8.

        A pool table has exactly one of each, so this is a choice across the
        whole set, not a test applied to each ball in isolation.  Classifying
        independently produced two cue balls and four 8 balls on real footage --
        grey cloth pushes several balls into "dark and colourless" at once.

        Working from the tracks' smoothed colour signatures keeps the
        assignment stable from frame to frame.
        """
        for track in self.tracks:
            track.role = None

        # Coasting tracks take part so that the ball holding a role keeps it
        # while it is briefly unseen; a newcomer must be confirmed (below).
        confirmed = [t for t in self.tracks if t.is_visible]
        if not confirmed:
            return

        def cue_like(t: Track, bar: float = 1.0) -> bool:
            L, C = t.signature.lightness_chroma
            return L >= _MIN_CUE_LIGHTNESS * bar and C <= _MAX_CUE_CHROMA / bar

        def eight_like(t: Track, bar: float = 1.0) -> bool:
            L, C = t.signature.lightness_chroma
            black = max(t.signature.dark_fraction, 1.0 - L / 128.0)
            return black >= _MIN_EIGHT_BLACK * bar and C <= _MAX_EIGHT_CHROMA / bar

        cue = self._pick_role(confirmed, "cue", lambda t: t.signature.cue_score, cue_like)

        if not self.table.has_pockets:
            return  # carom: no 8 ball
        rest = [t for t in confirmed if t is not cue]
        self._pick_role(rest, "eight", lambda t: t.signature.eight_score, eight_like)

    def _pick_role(self, candidates, role, score, eligible):
        """Give ``role`` to the best-scoring eligible track, stickily.

        ``eligible(track, bar)`` is the test a newcomer must pass at ``bar`` =
        1; the current holder keeps the role at the looser ``_ROLE_HOLD_RATIO``.
        """
        held = self._role_holders.get(role)
        usable = [
            t for t in candidates
            if (t.track_id == held and eligible(t, _ROLE_HOLD_RATIO))
            or (t.state is TrackState.CONFIRMED and eligible(t))
        ]
        if not usable:
            # Remember who held it: a holder hidden for a while, or revived
            # from limbo, should get it back without re-qualifying from scratch.
            return None

        best = max(usable, key=score)
        incumbent = next((t for t in usable if t.track_id == held), None)
        if incumbent is not None and score(incumbent) >= score(best) - _ROLE_STICKINESS:
            best = incumbent

        best.role = role
        self._role_holders[role] = best.track_id
        return best

    def assign_numbers(self) -> None:
        """Name every coloured ball by its number, across the whole table.

        At most one ball can be each number, so this is one optimal assignment
        of balls to numbers rather than a guess per ball -- the same reasoning
        as ``assign_roles``, which has already picked the cue ball and the 8.
        A ball that no number fits keeps its track id.  Numbers held by a ball
        waiting in limbo, or by one that was potted, are not given to anyone
        else: that ball is still, or was last, that number.

        With ``balls.ball_set: auto`` both sets are fitted every frame and the
        evidence for each is accumulated; the set in use changes only when the
        other is clearly ahead, so the choice settles once a pink or an orange
        ball, or a stripe's caps, have been seen, and then does not flicker.
        Only balls whose colour was measured on their own count (see
        ``Track.clean_samples``): in the rack every ball samples its neighbours,
        and both synthetic clips chose the wrong set there before this.
        """
        bc = self.cfg.balls
        for track in self.tracks:
            if track.role is not None or not track.is_visible:
                track.number = None
        if not bc.enabled or not self.table.has_pockets:
            return
        candidates = [
            t for t in self.tracks
            if t.is_visible and t.role is None and t.clean_samples >= bc.min_colour_samples
        ]
        if not candidates:
            return
        taken = {
            t.number for t in list(self.limbo) + list(self.finished)
            if t.number is not None and (t in self.limbo or t.death_reason == "potted")
        }
        allowed = [n for n in self._allowed_numbers if n not in taken]
        observed = []
        for t in candidates:
            L, hue, C = ballnum.colour_terms(t.signature.ball_colour)
            observed.append((L, hue, C, t.signature.stripe, t.signature.dark_fraction))
        current = [t.number for t in candidates]

        results = {}
        for name in self._ball_sets:
            specs = ballnum.ball_specs(name, allowed)
            numbers, _, scales = ballnum.assign(
                observed, specs, bc.max_cost, current, bc.stickiness,
                *self._set_scales[name],
            )
            results[name] = numbers
            self._set_scales[name] = scales
            # Evidence for this set: the squared cost of every ball under it,
            # an unnumbered one counting as the most a number may cost.  A sum,
            # not a mean, and accumulated, because usually a single ball is
            # all that tells the two sets apart -- a pink or orange one, or a
            # stripe's caps -- and averaged over the whole table its voice
            # was lost: fedor_jump's 9, black-capped, was named the orange 13.
            raw = ballnum.cost_matrix(observed, specs, *scales)
            index = {s.number: j for j, s in enumerate(specs)}
            energy = float(sum(
                min(raw[i, index[n]], bc.max_cost) ** 2 if n is not None else bc.max_cost ** 2
                for i, n in enumerate(numbers)
            ))
            self._set_fit[name] = _SET_EVIDENCE_DECAY * self._set_fit.get(name, 0.0) + energy

        if len(self._ball_sets) > 1:
            best = min(self._ball_sets, key=lambda s: self._set_fit[s])
            if self.ball_set is None or (
                self._set_fit[self.ball_set] - self._set_fit[best] > _SET_SWITCH_MARGIN
            ):
                self.ball_set = best
        chosen = self.ball_set or self._ball_sets[0]
        for track, number in zip(candidates, results[chosen]):
            track.number = number

    def cue_ball(self) -> Optional[Track]:
        """The track currently holding the cue-ball role."""
        return next((t for t in self.active_tracks() if t.role == "cue"), None)

    @property
    def tracks_created(self) -> int:
        return self._next_id - 1

    def reset_trails(self) -> None:
        for track in self.tracks:
            track.trail.clear()
