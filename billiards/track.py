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
#: cost units: one ball fitting 2 units better for one frame is 4).  Only a
#: few balls tell the sets apart -- a pink one, an orange one, a stripe's
#: caps -- and once they are potted nothing does: remembered at 0.98 a frame,
#: the pink 4 of the 2026 Premier League final was forgotten two seconds after
#: it dropped, and the purple 5 became the 4 of the other set.
_SET_EVIDENCE_DECAY = 0.999
_SET_SWITCH_MARGIN = 6.0

#: A ball set aside at a cut and looked for in a camera view it has not been
#: seen in is compared with the colour another camera gave it, through a gate
#: this many times looser (``MultiObjectTracker._reclaim_cost``).
_CROSS_VIEW_COLOUR = 1.8

#: With the ball model, each ball's answers are averaged with this weight per
#: detection, once it has a few; a track is confirmed only if its average
#: chance of being a ball is at least ``_MODEL_CONFIRM``; and a ball is the
#: cue ball or the 8 only if the model gives it at least ``_MODEL_ROLE``
#: (the holder keeps it down to ``_ROLE_HOLD_RATIO`` of that).
_MODEL_ALPHA = 0.06
_MODEL_CONFIRM = 0.5
_MODEL_ROLE = 0.5

#: How much better a rival has to score before a role changes hands.  Two
#: similar-looking balls otherwise trade the "CUE" label back and forth every
#: few frames, which is worse than being slightly wrong consistently.
_ROLE_STICKINESS = 0.10


def balls_model_black() -> int:
    """Where black is among the ball model's colour families."""
    return ballnum.MODEL_FAMILIES.index("black")


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
        "aside_frame", "aside_frames", "view_colour",
        "ball_evidence", "cue_evidence", "family_logp", "stripe_evidence", "model_samples",
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
        #: While set aside at a camera cut (``MultiObjectTracker.set_aside``):
        #: the frame it was, and how many frames with the table in view it
        #: has waited since.
        self.aside_frame: Optional[int] = None
        self.aside_frames = 0
        #: What this ball looks like from each camera view
        #: (``MultiObjectTracker.set_view``).  Two cameras do not show a ball
        #: in the same colour: on the 2026 US Open the same balls, found again
        #: 0.3-3 in from where they were after a cut, differed by 45-104 in
        #: colour distance from what the other camera had shown, against a
        #: gate of 42, and were taken for new balls.
        self.view_colour: Dict[int, ColorSignature] = {}
        #: What the ball model (``billiards.ballnet``) has made of this ball,
        #: averaged over its detections: the chance it is a ball at all, and --
        #: over detections of it on its own -- that it is the cue ball, the
        #: log chance of each colour family, the chance it is a stripe.  None
        #: without the model.
        self.ball_evidence: Optional[float] = None
        self.cue_evidence: Optional[float] = None
        self.family_logp: Optional[np.ndarray] = None
        self.stripe_evidence: Optional[float] = None
        self.model_samples = 0
        self._learn_model(detection)

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

    def _learn_model(self, detection: Detection) -> None:
        """Fold one detection's ball-model answers into this ball's."""
        if detection.family_p is None:
            return
        n_all = self.hits
        a = max(_MODEL_ALPHA, 1.0 / max(1, n_all))
        self.ball_evidence = detection.ball_p if self.ball_evidence is None else (
            (1.0 - a) * self.ball_evidence + a * detection.ball_p)
        if not detection.colour_ok:
            return
        self.model_samples += 1
        a = max(_MODEL_ALPHA, 1.0 / self.model_samples)
        logp = np.log(np.clip(detection.family_p, 1e-4, 1.0))
        if self.family_logp is None:
            self.cue_evidence, self.family_logp, self.stripe_evidence = detection.cue_p, logp, detection.stripe_p
        else:
            self.cue_evidence = (1.0 - a) * self.cue_evidence + a * detection.cue_p
            self.family_logp = (1.0 - a) * self.family_logp + a * logp
            self.stripe_evidence = (1.0 - a) * self.stripe_evidence + a * detection.stripe_p

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, cfg: Config, view: int = 0) -> None:
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
        #
        # Learning a new camera's colour quickly instead (as fast as a new
        # ball's) was tried on 28 Sep 2026: a broadcast camera that pushes in
        # or pans becomes a new view each time (2026 Premier League final:
        # eight), and balls relearning their colour at every one traded
        # identities -- 1.9 to 2.6 ids per ball against the answer key.
        if detection.colour_ok:
            self.clean_samples += 1
            if self.clean_samples == 1:
                self.signature = detection.signature
            else:
                alpha = max(0.08, 1.0 / self.clean_samples)
                self.signature = self.signature.blend(detection.signature, alpha)
            self.view_colour[view] = self.signature
        self._learn_model(detection)

        if self.state is TrackState.TENTATIVE:
            # Seen mostly past the far edge, it never rolled there from the
            # bed: a hand on the rail, or a ball's own top in the shadow under
            # the cushion's nose.  It is never confirmed.  Nor is one the ball
            # model mostly thinks is not a ball.
            if (
                self.hits >= cfg.tracker.min_hits_to_confirm
                and 2 * self.band_hits <= self.hits
                and (self.ball_evidence is None or self.ball_evidence >= _MODEL_CONFIRM)
            ):
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
        #: Balls lost -- out of limbo, not potted -- waiting longer still, for
        #: a ball just like them where they were (``_revive``).
        self.gone: List[Track] = []
        self._gone_frames = int(round(cfg.tracker.lost_revive_window_s * self.fps))
        #: Balls set aside at a camera cut, waiting to be seen again.  See
        #: ``set_aside`` and ``_reclaim``.
        self.aside: List[Track] = []
        self.reclaimed = 0
        #: The camera view being tracked in (``set_view``).
        self.view = 0
        self._aside_frames = max(1, int(round(cfg.tracker.reclaim_window_s * self.fps)))
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

        confirmed_now: List[Track] = []
        for ti, di in matches:
            track = self.tracks[ti]
            was_tentative = track.state is TrackState.TENTATIVE
            track.update(detections[di], self.cfg, self.view)
            if was_tentative and track.state is TrackState.CONFIRMED:
                confirmed_now.append(track)

        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed(self.cfg)
            # Only a ball that was not seen: a seen one is where it was seen.
            self._bounce_off_rails(self.tracks[ti])

        fresh = [di for di in unmatched_dets if not self._revive(detections[di], frame)]
        if self.aside:
            fresh = self._reclaim(detections, fresh, frame)
        for di in fresh:
            # Past the far edge only balls already being followed are looked
            # for: a hand on the far rail is there too, and is never a ball
            # that rolled there.
            if not detections[di].in_raised_band:
                self._spawn(detections[di], frame)
        if self.aside and confirmed_now:
            # Hidden (by a player, a cluster) on the first frames back, a ball
            # set aside at the cut is found by its new track instead.
            self._absorb(confirmed_now)

        self._retire(frame)
        self._expire_limbo(frame)
        self._expire_aside(frame)
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
        if detection.colour_ok:
            track.view_colour[self.view] = track.signature
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
        if not self.limbo and not self.gone:
            return False
        tc = self.cfg.tracker
        best = self._revivable(self.limbo, detection, frame, tc.revive_distance_ball_diameters, True)
        if best is not None:
            self.limbo.remove(best)
        else:
            # A ball lost for longer is looked for only where it was, and only
            # in something the ball model, if it looked, takes for a ball.
            if detection.ball_p < _MODEL_CONFIRM:
                return False
            best = self._revivable(self.gone, detection, frame, tc.lost_revive_distance_ball_diameters, False)
            if best is None:
                return False
            self.gone.remove(best)
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
        best.update(detection, self.cfg, self.view)
        self.tracks.append(best)
        self.revived += 1
        return True

    def _revivable(self, waiting: Sequence[Track], detection: Detection, frame: int,
                   reach_diameters: float, rolling: bool) -> Optional[Track]:
        """The ball among ``waiting`` this detection is, if any."""
        if not waiting:
            return None
        tc = self.cfg.tracker
        det = np.asarray(detection.centre_table, dtype=np.float64)
        colours = colour_distance_matrix([t.signature for t in waiting], [detection.signature])[:, 0]
        best, best_cost = None, None
        for track, colour in zip(waiting, colours):
            # A ball that vanished while rolling -- off the far rail, where it
            # cannot be seen -- turns up further along, so the reach grows with
            # how fast it was going and how long it has been gone.
            gone_s = max(0, frame - (track.death_frame or frame)) / self.fps
            reach = reach_diameters * self.table.ball_diameter_in + (track.speed * gone_s if rolling else 0.0)
            gap = float(np.linalg.norm(np.asarray(track.last_observed_xy) - det))
            if gap > reach or colour > tc.max_color_distance:
                continue
            cost = gap / reach + colour / tc.max_color_distance
            if best_cost is None or cost < best_cost:
                best, best_cost = track, cost
        return best

    # -- camera cuts ------------------------------------------------------

    def set_view(self, view: int) -> None:
        """Track in camera view ``view`` from now on.

        Each ball takes the colour it has in that camera, if it has been seen
        there; otherwise it keeps the last camera's, and is looked for more
        loosely when it is set aside (``_reclaim_cost``).
        """
        if view == self.view:
            return
        for track in list(self.tracks) + list(self.aside) + list(self.limbo) + list(self.gone):
            own = track.view_colour.get(view)
            if own is not None:
                track.signature = own
        self.view = view

    def set_aside(self, frame: int) -> None:
        """The camera has cut away from the table: set every ball aside.

        Nothing is seen until the table is back, and then possibly from
        another camera, after balls have rolled.  Tracked on, each ball was
        coasted, lost, and replaced by a new one when it was seen again --
        or, if it happened to coast near a pocket, reported potted.  Set
        aside, a ball keeps its identity until ``_reclaim`` finds it again.
        A tentative track was never a ball anyone saw, and is dropped.
        """
        for track in self.tracks:
            if track.state is TrackState.TENTATIVE:
                track.kill(frame, "spurious")
                self.finished.append(track)
                continue
            # A path drawn in one camera's picture means nothing in the next.
            track.trail.clear()
            track.aside_frame = frame
            track.aside_frames = 0
            self.aside.append(track)
        self.tracks = []

    def reference_balls(self) -> List[Track]:
        """The balls the table is known to hold: followed, or set aside."""
        return [t for t in self.tracks if t.is_visible] + list(self.aside)

    def expected_position(self, track: Track) -> np.ndarray:
        """Where a ball should be now: its filter's position for a ball set
        aside (rolled on through the cut, see ``coast_aside``), its last
        sighting otherwise."""
        if track.aside_frame is not None:
            return np.asarray(track.kf.position, dtype=np.float64)
        return np.asarray(track.last_observed_xy, dtype=np.float64)

    def coast_aside(self, dt: float) -> None:
        """Roll the balls set aside on through a frame the table is not seen.

        A broadcast cuts away mid-shot as readily as between shots, and a
        ball that was rolling when it did is not where it was last seen when
        the table is back: after the break in the synthetic broadcast clip,
        most were a foot or more away 0.8 s later.  Its motion model knows
        where it went -- slowing, and off the cushions -- well enough to find
        it again there.
        """
        rest = self.cfg.tracker.stationary_speed_in_s
        for track in self.aside:
            if track.speed < rest:
                continue
            track.kf.predict(dt)
            self._bounce_off_rails(track)

    def _reclaim_cost(self, xy: np.ndarray, signatures: Sequence[ColorSignature]) -> np.ndarray:
        """Cost of each ball set aside (rows) being each observation (columns).

        A ball found within ``reclaim_distance`` of where it was last seen,
        looking like it, is that ball: most balls do not move while the camera
        is away.  One that rolled is found by its colour alone, but only
        where that is unambiguous: close to it (``reclaim_colour_ratio``), and
        the closest by a margin of every ball set aside.  Otherwise it is left
        for a new track, which is the old behaviour, and no worse.
        """
        tc = self.cfg.tracker
        n, m = len(self.aside), len(xy)
        cost = np.full((n, m), FORBIDDEN, dtype=np.float64)
        if n == 0 or m == 0:
            return cost
        near = tc.reclaim_distance_ball_diameters * self.table.ball_diameter_in
        cmax = tc.max_color_distance
        last = np.array([self.expected_position(t) for t in self.aside], dtype=np.float64).reshape(n, 2)
        # A ball that rolled on through the cut is where its motion model put
        # it, give or take a fifth of how far it went.
        rolled = np.array([
            np.linalg.norm(self.expected_position(t) - np.asarray(t.last_observed_xy)) for t in self.aside
        ])
        near = near + 0.2 * rolled[:, None]
        dist = np.linalg.norm(last[:, None, :] - np.asarray(xy, dtype=np.float64).reshape(m, 2)[None], axis=2)
        colour = colour_distance_matrix([t.signature for t in self.aside], signatures)
        # A ball not yet seen from this camera is compared with the colour
        # another camera gave it, and more loosely.
        gates = np.array([
            cmax if self.view in t.view_colour else _CROSS_VIEW_COLOUR * cmax for t in self.aside
        ])[:, None]

        near_ok = (dist <= near) & (colour <= gates)
        cost[near_ok] = (dist / near + colour / gates)[near_ok]

        ordered = np.sort(colour, axis=0)
        runner_up = ordered[1] if n > 1 else np.full(m, np.inf)
        margin = 0.25 * cmax
        far_ok = (
            ~near_ok
            & (colour <= tc.reclaim_colour_ratio * cmax)
            & (colour <= ordered[0][None, :])
            & (runner_up[None, :] >= colour + margin)
        )
        cost[far_ok] = (2.0 + colour / cmax)[far_ok]
        return cost

    def _give_back(self, track: Track, detection_xy, frame: int) -> None:
        """Return a ball set aside to the tracker, at ``detection_xy``."""
        tc = self.cfg.tracker
        self.aside.remove(track)
        track.kf = BallKalman(
            detection_xy,
            velocity_tau_s=tc.velocity_tau_s,
            accel_std_in_s2=tc.accel_std_in_s2,
            meas_std_in=tc.meas_std_in,
            init_vel_std_in_s=tc.init_vel_std_in_s,
            manoeuvre_gain_max=tc.manoeuvre_gain_max,
        )
        track.state = TrackState.CONFIRMED
        track.aside_frame = None
        track.aside_frames = 0
        self.tracks.append(track)
        self.reclaimed += 1

    def _reclaim(self, detections: Sequence[Detection], candidates: List[int], frame: int) -> List[int]:
        """Give detections that match no followed ball to balls set aside.

        One optimal assignment over all of them at once, as for tracking: the
        first frame back from a cut matches every ball on the table here.
        Returns the detections that matched none.
        """
        usable = [di for di in candidates if not detections[di].in_raised_band]
        if not usable:
            return candidates
        dets = [detections[di] for di in usable]
        cost = self._reclaim_cost(
            np.array([d.centre_table for d in dets], dtype=np.float64).reshape(-1, 2),
            [d.signature for d in dets],
        )
        matches, _, _ = associate(cost)
        taken = set()
        for ai, j in sorted(matches, key=lambda p: -p[0]):
            track = self.aside[ai]
            new_camera = self.view not in track.view_colour
            self._give_back(track, dets[j].centre_table, frame)
            if new_camera and dets[j].colour_ok:
                # Found in a camera it has not been seen in: this is what it
                # looks like here, and what it is followed by from now on.
                track.signature = dets[j].signature
            track.update(dets[j], self.cfg, self.view)
            taken.add(usable[j])
        return [di for di in candidates if di not in taken]

    def _absorb(self, newcomers: Sequence[Track]) -> None:
        """New tracks, just confirmed, that are balls set aside at a cut."""
        if not self.aside or not newcomers:
            return
        cost = self._reclaim_cost(
            np.array([t.kf.position for t in newcomers], dtype=np.float64).reshape(-1, 2),
            [t.signature for t in newcomers],
        )
        matches, _, _ = associate(cost)
        for ai, j in sorted(matches, key=lambda p: -p[0]):
            old, new = self.aside[ai], newcomers[j]
            self.aside.remove(old)
            if self.view not in old.view_colour and new.clean_samples:
                old.signature = new.signature
                old.view_colour[self.view] = new.signature
            # The ball keeps its name, and takes over what its new track has
            # measured since: the filter, the path, the last sighting.
            old.kf = new.kf
            old.state = TrackState.CONFIRMED
            old.hits += new.hits
            old.time_since_update = new.time_since_update
            old.last_image_xy, old.last_observed_xy = new.last_image_xy, new.last_observed_xy
            old.last_radius_px = new.last_radius_px
            old.trail.clear()
            old.trail.extend(new.trail)
            old.aside_frame, old.aside_frames = None, 0
            self.tracks[self.tracks.index(new)] = old
            self.reclaimed += 1

    def _expire_aside(self, frame: int) -> None:
        """Give up on balls set aside that have not been seen again.

        Only frames with the table in view count (this runs only then): a
        close-up of a player can outlast any window, and the balls are still
        on the table.  A ball heading into a pocket when the camera cut away
        was potted, dated to the cut; any other is lost.
        """
        keep: List[Track] = []
        for track in self.aside:
            track.aside_frames += 1
            if track.aside_frames < self._aside_frames:
                keep.append(track)
                continue
            pos, vel = track.kf.position, track.kf.velocity
            potted = self._heading_into_pocket(np.asarray(track.last_observed_xy), vel) or (
                self.table.nearest_pocket_distance((float(pos[0]), float(pos[1])))
                <= self.cfg.events.pocket_radius_ball_diameters * self.table.ball_diameter_in
            )
            track.kill(track.aside_frame if track.aside_frame is not None else frame,
                       "potted" if potted and track.speed > self.cfg.tracker.stationary_speed_in_s else "lost")
            (self.gone if track.death_reason == "lost" and self._gone_frames > 0 else self.finished).append(track)
        self.aside = keep

    def _expire_limbo(self, frame: int) -> None:
        keep: List[Track] = []
        for track in self.limbo:
            if track.death_frame is not None and frame - track.death_frame >= self._limbo_frames:
                (self.gone if track.death_reason == "lost" and self._gone_frames > 0 else self.finished).append(track)
            else:
                keep.append(track)
        self.limbo = keep
        keep = []
        for track in self.gone:
            if track.death_frame is not None and frame - track.death_frame >= self._gone_frames:
                self.finished.append(track)
            else:
                keep.append(track)
        self.gone = keep

    def flush_limbo(self) -> None:
        """End of clip: nothing more can come back."""
        self.finished.extend(self.limbo)
        self.finished.extend(self.gone)
        self.limbo = []
        self.gone = []

    def flush_aside(self, frame: int) -> None:
        """End of clip: balls still set aside at a cut were not seen again."""
        for track in self.aside:
            track.kill(track.aside_frame if track.aside_frame is not None else frame, "lost")
            self.finished.append(track)
        self.aside = []

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
        for track in list(self.tracks) + list(self.limbo) + list(self.aside) + list(self.gone):
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
        return list(self.tracks) + list(self.aside) + list(self.limbo) + list(self.gone) + list(self.finished)

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

        if all(t.family_logp is not None for t in confirmed):
            # The ball model's word for it, where every ball has one.
            black = balls_model_black()

            def cue_score(t: Track) -> float:
                return float(t.cue_evidence)

            def eight_score(t: Track) -> float:
                return float(np.exp(t.family_logp[black])) * (1.0 - float(t.cue_evidence))

            def cue_like(t: Track, bar: float = 1.0) -> bool:
                return cue_score(t) >= _MODEL_ROLE * bar

            def eight_like(t: Track, bar: float = 1.0) -> bool:
                return eight_score(t) >= _MODEL_ROLE * bar
        else:
            def cue_score(t: Track) -> float:
                return t.signature.cue_score

            def eight_score(t: Track) -> float:
                return t.signature.eight_score

            def cue_like(t: Track, bar: float = 1.0) -> bool:
                L, C = t.signature.lightness_chroma
                return L >= _MIN_CUE_LIGHTNESS * bar and C <= _MAX_CUE_CHROMA / bar

            def eight_like(t: Track, bar: float = 1.0) -> bool:
                L, C = t.signature.lightness_chroma
                black = max(t.signature.dark_fraction, 1.0 - L / 128.0)
                return black >= _MIN_EIGHT_BLACK * bar and C <= _MAX_EIGHT_CHROMA / bar

        cue = self._pick_role(confirmed, "cue", cue_score, cue_like)

        if not self.table.has_pockets:
            return  # carom: no 8 ball
        rest = [t for t in confirmed if t is not cue]
        self._pick_role(rest, "eight", eight_score, eight_like)

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
        # A ball set aside at a cut keeps its number too, which is also what
        # lets it be found again by colour rather than as a rival 7.  A potted
        # ball's number is kept from the rest, unless the ball model is naming
        # them: then a ball of that colour on the table is that ball -- the pot
        # was wrong, or it is a new rack (the 2026 US Open highlights run two
        # racks, and every ball potted in the first went unnamed in the second).
        modelled = any(t.family_logp is not None for t in self.tracks)
        taken = {
            t.number for t in list(self.limbo) + list(self.finished) + list(self.aside)
            if t.number is not None
            and (t in self.limbo or t in self.aside or (t.death_reason == "potted" and not modelled))
        }
        allowed = [n for n in self._allowed_numbers if n not in taken]
        current = [t.number for t in candidates]
        # The ball model's colour and pattern, where every ball has them
        # (``ballnum.model_cost_matrix``); otherwise the colour measured off
        # the picture, against a palette (``ballnum.cost_matrix``).
        by_model = all(t.family_logp is not None for t in candidates)
        if by_model:
            fam = [t.family_logp for t in candidates]
            stripe = [t.stripe_evidence for t in candidates]
            max_cost = bc.model_max_cost
        else:
            observed = []
            for t in candidates:
                L, hue, C = ballnum.colour_terms(t.signature.ball_colour)
                observed.append((L, hue, C, t.signature.stripe, t.signature.dark_fraction))
            max_cost = bc.max_cost

        results = {}
        for name in self._ball_sets:
            specs = ballnum.ball_specs(name, allowed)
            if by_model:
                raw = ballnum.model_cost_matrix(fam, stripe, specs, [t.signature.dark_fraction for t in candidates])
                numbers = ballnum.assign_costs(raw, specs, max_cost, current, bc.model_stickiness)
            else:
                numbers, _, scales = ballnum.assign(
                    observed, specs, bc.max_cost, current, bc.stickiness,
                    *self._set_scales[name],
                )
                self._set_scales[name] = scales
                raw = ballnum.cost_matrix(observed, specs, *scales)
            results[name] = numbers
            # Evidence for this set: the squared cost of every ball under it,
            # an unnumbered one counting as the most a number may cost.  A sum,
            # not a mean, and accumulated, because usually a single ball is
            # all that tells the two sets apart -- a pink or orange one, or a
            # stripe's caps -- and averaged over the whole table its voice
            # was lost: fedor_jump's 9, black-capped, was named the orange 13.
            index = {s.number: j for j, s in enumerate(specs)}
            energy = float(sum(
                min(raw[i, index[n]], max_cost) ** 2 if n is not None else max_cost ** 2
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
