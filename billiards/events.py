"""Shot events: ball-ball collisions, cushion contacts, pots and balls struck.

The old collision test was ``np.linalg.norm(cue - object) < 20`` -- twenty
pixels.  Twenty pixels is about one ball at 480p on a tight camera, three balls
on a wide one, and a third of a ball at 1080p, which is why the contact marker
had to be re-tuned per clip and still fired on balls that never touched.

Here contact is "closer than 1.12 ball diameters **and** actually closing on
each other".  Both quantities are physical, so the same rule holds for every
video, every resolution and every camera angle.  Requiring a positive closing
speed is what stops two balls resting against each other from emitting a
collision on every single frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .geometry import TableModel
from .track import Track, TrackState


#: Frames a track must have existed before its speed is trusted enough to say
#: it was struck.  The filter starts with a wide velocity prior by design.
_MIN_AGE_FOR_STRUCK = 8

#: A ball at rest seen next more than a ball's width away, within this many
#: seconds and trail samples (``EventDetector._jump_speed``), went at that
#: step's speed: longer, and it may have been carried there by hand.
_JUMP_WINDOW_S = 0.15
_JUMP_MAX_SAMPLES = 4

#: A contact with a ball at rest is reported once that ball has gone this
#: many ball diameters, and dropped if it has not within this many seconds
#: and is still in sight (``EventDetector._confirm_contacts``).
_CONTACT_MOVE_DIAMETERS = 0.5
_CONTACT_CONFIRM_S = 0.3

#: Balls set moving on the same frame farther apart than this, in ball
#: diameters, with nothing else moving, were not struck
#: (``EventDetector._picture_moved``): a ball frozen to the cue ball is set
#: off with it, within a diameter.
_TOGETHER_BALL_DIAMETERS = 3.0

#: Seconds after the table comes back from a cut in which no ball is called
#: struck (see ``EventDetector.forget``): long enough for a ball picked up
#: mid-roll to have its speed measured.
_QUIET_AFTER_CUT_S = 0.3


def closest_approach(
    gap: np.ndarray, change: np.ndarray
) -> Tuple[float, float]:
    """How near two balls came during one frame, and when.

    ``gap`` is the vector between them at the start of the frame and ``change``
    is how that vector changed by the end of it.  Both balls travel in a
    straight line over so short an interval, so the distance between them is
    ``|gap + s * change|`` for ``s`` in 0..1 and its minimum is exact.

    Returns ``(s, distance)`` at that minimum.
    """
    speed_sq = float(np.dot(change, change))
    if speed_sq <= 1e-12:  # not moving relative to each other
        return 0.0, float(np.linalg.norm(gap))
    s = float(np.clip(-np.dot(gap, change) / speed_sq, 0.0, 1.0))
    return s, float(np.linalg.norm(gap + s * change))


class EventType(Enum):
    COLLISION = "collision"
    CUSHION = "cushion"
    POT = "pot"
    BALL_STRUCK = "ball_struck"


@dataclass
class Event:
    type: EventType
    frame: int
    t_s: float
    table_xy: Tuple[float, float]
    image_xy: Tuple[float, float]
    track_ids: Tuple[int, ...]
    detail: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "frame": self.frame,
            "t_s": round(self.t_s, 3),
            "x_in": round(self.table_xy[0], 3),
            "y_in": round(self.table_xy[1], 3),
            "x_px": round(self.image_xy[0], 1),
            "y_px": round(self.image_xy[1], 1),
            "track_ids": list(self.track_ids),
            **{k: round(float(v), 3) for k, v in self.detail.items()},
        }


@dataclass
class Kink:
    """A ball's path changing from one straight line to another.

    Found from the raw detections, not the filter: the filter's velocity turns
    a corner over two or three frames, so by the time it has visibly reversed
    the ball is several inches off the cushion it bounced from.  The two raw
    straight lines, extended, still meet exactly where the contact happened --
    even across frames where the ball was not seen at all.
    """

    track_id: int
    t: float  #: scene time of the kink
    point: np.ndarray  #: where the two lines meet, table inches
    v_in: np.ndarray
    v_out: np.ndarray


def _fit_line(
    samples: Sequence[Tuple[float, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares ``p(t) = a + v t``; returns ``(a, v, max residual, SSE)``."""
    t = np.array([s[0] for s in samples], dtype=np.float64)
    p = np.array([s[1] for s in samples], dtype=np.float64)
    t0 = t.mean()
    dt = t - t0
    denom = float(np.dot(dt, dt))
    v = (dt @ (p - p.mean(axis=0))) / denom if denom > 1e-12 else np.zeros(2)
    a = p.mean(axis=0) - v * t0
    resid = np.linalg.norm(p - (a + np.outer(t, v)), axis=1)
    return a, v, float(np.max(resid)), float(np.dot(resid, resid))


#: Samples in each of the two straight lines either side of a corner.
_KINK_SIDE = 3


def find_kink(
    history: Sequence[Tuple[float, np.ndarray]],
    min_change_in_s: float,
    after_t: float = -np.inf,
    max_residual_in: float = 0.6,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """The corner in a ball's recent path, once it is certain where it is.

    ``history`` is ``(scene time, position)`` of consecutive observations; only
    those after ``after_t`` (the previous corner) are used, so a corner is never
    fitted across.  Every split into three samples in and three out is tried,
    and the one whose two lines are straightest wins -- but it is only
    returned once the split *after* it has also been tried and lost.

    That last condition is the point.  Accepting the first split whose two
    lines disagree enough -- as an earlier version did -- calls the corner a
    frame early, with the outgoing line straddling the bounce: half in, half
    out, so the motion toward the rail looks merely slowed rather than
    reversed.  A slow ball barely bends such a line, so no straightness test
    catches it.  Half the cushion contacts on the synthetic break were lost
    that way.  Returns ``(t, point, v_in, v_out)`` or None.
    """
    h = [s for s in history if s[0] > after_t]
    n = len(h)
    k = _KINK_SIDE
    best = None
    for b in range(k, n - k + 1):
        pre, post = h[b - k:b], h[b:b + k]
        a_in, v_in, r_in, sse_in = _fit_line(pre)
        a_out, v_out, r_out, sse_out = _fit_line(post)
        size = float(np.linalg.norm(v_out - v_in))
        fastest = max(float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out)))
        # Relative as well as absolute: friction and noise change a fast
        # ball's velocity by more than a slow one's with nothing happening.
        if size < max(min_change_in_s, 0.25 * fastest):
            continue
        cost = sse_in + sse_out
        if best is None or cost < best[0]:
            best = (cost, b, a_in, v_in, a_out, v_out, max(r_in, r_out))
    if best is None:
        return None
    _, b, a_in, v_in, a_out, v_out, resid = best
    if b > n - k - 1 or resid > max_residual_in:
        # The latest split cannot yet have been beaten by the one after it,
        # or the lines are not straight enough to trust.
        return None

    # When the two lines are closest.  A real corner is between the last
    # incoming sample and the first outgoing one; clamping keeps a noisy fit
    # from placing it anywhere else.
    change = v_in - v_out
    t = -float(np.dot(a_in - a_out, change)) / float(np.dot(change, change))
    t = float(np.clip(t, h[b - 1][0], h[b][0]))
    point = ((a_in + v_in * t) + (a_out + v_out * t)) / 2.0
    return t, point, v_in, v_out


class EventDetector:
    def __init__(self, cfg: Config, table: TableModel, fps: float = 30.0) -> None:
        self.cfg = cfg
        self.table = table
        self.fps = max(float(fps), 1.0)
        self.events: List[Event] = []
        self._last_pair_event: Dict[frozenset, float] = {}
        self._last_cushion: Dict[int, float] = {}
        self._prev_velocity: Dict[int, np.ndarray] = {}
        self._prev_position: Dict[int, np.ndarray] = {}
        #: Per-track motion state, "rest" or "moving".  Not a plain boolean:
        #: see _balls_struck for why it needs a hysteresis band.
        self._motion_state: Dict[int, str] = {}

        #: Scene time, advanced by each step's ``dt``, and which file frame and
        #: timestamp each measured step was, so a kink found a few frames late
        #: can be reported at the frame where it happened.
        self._t_scene = 0.0
        self._frame_clock: Deque[Tuple[float, int, float]] = deque(maxlen=256)
        #: Raw observed positions per track, and when its last kink was.
        self._observed: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}
        self._last_kink_t: Dict[int, float] = {}
        #: Kinks not (yet) explained, waiting for a partner to make a collision.
        self._pending: List[Kink] = []
        #: Per track, per rail: whether its last decisive raw step was toward
        #: or away from that rail, and where.  See ``_rail_reversal``.
        self._rail_state: Dict[int, Dict[int, tuple]] = {}
        #: Which detectors run.  Measured on synthetic ground truth and the
        #: sample clips (see DIARY.md, 23 Sep): cushions come from the path
        #: (``_rail_reversal`` and corners), collisions from both the path and
        #: the sampled closest approach, which finds contacts where a ball is
        #: potted or hidden too soon after for its path to turn a corner.
        self.kink_events = True
        self.kink_cushions = True
        self.sampled_events = True
        #: Scene time until which no ball is called struck (``forget``).
        self._quiet_until = -np.inf
        #: Frames on which balls seemed struck together (``_picture_moved``).
        self.picture_moves = 0
        #: Contacts with a ball at rest, held until it moves
        #: (``_confirm_contacts``): (event, that ball, where it was, deadline).
        self._held: List[Tuple[Event, int, np.ndarray, float]] = []
        #: Pairs of balls in contact on this step, reported or held: the
        #: filters are told either way (``TrackingPipeline.process``).
        self.contacts_now: List[Tuple[int, ...]] = []

    @property
    def _struck_speed(self) -> float:
        """Speed above which a ball is definitely moving under its own steam."""
        return max(self.cfg.tracker.stationary_speed_in_s * 6.0, 12.0)

    @property
    def _rest_speed(self) -> float:
        """Speed below which a ball is definitely at rest."""
        return self.cfg.tracker.stationary_speed_in_s

    # -- public ------------------------------------------------------------

    def step(
        self,
        tracks: Sequence[Track],
        frame: int,
        t_s: float,
        dt: Optional[float] = None,
    ) -> List[Event]:
        """Events since the previous call, ``dt`` seconds ago.

        ``dt`` is passed rather than assumed to be ``1 / fps`` because it is
        not: the pipeline skips frames that merely repeat the one before them,
        so two calls can be two or three frame intervals apart.  Both the
        contact test and the cushion test measure how far a ball travelled
        since the last call, and getting that wrong by 2x is the difference
        between finding the contact and stepping over it.
        """
        dt = 1.0 / self.fps if dt is None else max(float(dt), 1e-4)
        self._t_scene += dt
        self._frame_clock.append((self._t_scene, frame, t_s))
        confirmed = [t for t in tracks if t.state is TrackState.CONFIRMED]
        new: List[Event] = []
        if self.kink_events:
            new.extend(self._kink_events(confirmed))
        self.contacts_now = []
        if self.sampled_events:
            new.extend(self._collisions(confirmed, frame, t_s, dt))
        new.extend(self._confirm_contacts(confirmed))
        struck = self._balls_struck(confirmed, frame, t_s)
        if self._picture_moved(struck, confirmed):
            # Not play: see ``_picture_moved``.  Nothing struck, and no
            # contact between balls that only seemed to move.
            self.picture_moves += 1
            new = [e for e in new if e.type is not EventType.COLLISION]
        else:
            new.extend(struck)

        for track in confirmed:
            self._prev_velocity[track.track_id] = track.velocity.copy()
            self._prev_position[track.track_id] = track.kf.position.copy()

        self.events.extend(new)
        return new

    def forget(self) -> None:
        """The camera cut away: every ball's path so far ends here.

        When the table is back a ball is picked up where it is then, which
        may be feet from where it was.  Joined to its old path, that jump
        is a corner, a bounce, a strike.  So the paths start again, and for a
        moment nothing is called struck: a ball picked up mid-roll has a
        filter that starts at rest, and would read as struck when it caught up.
        """
        self._prev_velocity.clear()
        self._prev_position.clear()
        self._motion_state.clear()
        self._observed.clear()
        self._last_kink_t.clear()
        self._pending.clear()
        self._held.clear()
        self._rail_state.clear()
        self._quiet_until = self._t_scene + _QUIET_AFTER_CUT_S

    def note_pot(self, track: Track, frame: int, t_s: float) -> Event:
        pos = track.kf.position
        img = self.table.ball_table_to_image([tuple(pos)])[0]
        event = Event(
            type=EventType.POT,
            frame=frame,
            t_s=t_s,
            table_xy=(float(pos[0]), float(pos[1])),
            image_xy=(float(img[0]), float(img[1])),
            track_ids=(track.track_id,),
            detail={"pocket_distance_in": self.table.nearest_pocket_distance(
                (float(pos[0]), float(pos[1]))
            )},
        )
        self.events.append(event)
        return event

    # -- kinks -------------------------------------------------------------

    def _frame_at(self, t_scene: float) -> Tuple[int, float]:
        """The first measured frame whose scene time is at or after ``t_scene``."""
        for t, frame, t_s in self._frame_clock:
            if t >= t_scene - 1e-9:
                return frame, t_s
        _, frame, t_s = self._frame_clock[-1]
        return frame, t_s

    def _kink_events(self, tracks: Sequence[Track]) -> List[Event]:
        """Cushions and collisions, from corners in the balls' raw paths.

        * A corner one ball radius off a rail, where the motion toward that
          rail turned into motion away from it, is a **cushion** contact.
        * Two balls turning a corner at the same moment, a ball's width apart,
          while closing on each other, is a **collision**.

        Both are reported a couple of frames after they happen -- the outgoing
        line needs two points -- but at the frame where they happened.
        """
        cfg = self.cfg.events
        out: List[Event] = []
        for track in tracks:
            if track.time_since_update != 0:
                continue
            history = self._observed.setdefault(track.track_id, deque(maxlen=10))
            history.append((self._t_scene, np.asarray(track.last_observed_xy, dtype=np.float64)))

            bounce = self._rail_reversal(track.track_id, history)
            if bounce is not None:
                out.append(bounce)

            last = self._last_kink_t.get(track.track_id, -np.inf)
            found = find_kink(history, cfg.min_bounce_speed_change_in_s, after_t=last)
            if found is None:
                continue
            t, point, v_in, v_out = found
            if t - last < cfg.kink_refractory_s:
                continue
            self._last_kink_t[track.track_id] = t
            kink = Kink(track.track_id, t, point, v_in, v_out)

            cushion = self._kink_cushion(kink) if self.kink_cushions else None
            if cushion is not None:
                out.append(cushion)
                continue
            partner = self._kink_partner(kink)
            if partner is not None:
                collision = self._kink_collision(partner, kink)
                if collision is not None:
                    out.append(collision)
                continue
            self._pending.append(kink)

        # A kink nobody else's kink explained.  The other ball may simply not
        # have shown a clean corner of its own -- it was at rest and its track
        # still new, it was hidden a frame after contact, it barely moved -- so
        # before giving up, look for a ball that was a ball's width away at
        # that moment.  A corner is still required, which is what keeps a near
        # miss from counting: two balls that pass close without touching do
        # not deflect.
        horizon = self._t_scene - 3.0 * cfg.kink_pair_window_s
        keep: List[Kink] = []
        for kink in self._pending:
            if kink.t >= horizon:
                keep.append(kink)
                continue
            collision = self._kink_single(kink)
            if collision is not None:
                out.append(collision)
        self._pending = keep
        return out

    def _position_at(self, track_id: int, t: float) -> Optional[np.ndarray]:
        """Where a track was observed to be at scene time ``t``, interpolated."""
        history = self._observed.get(track_id)
        if not history:
            return None
        samples = list(history)
        if t <= samples[0][0]:
            return samples[0][1] if samples[0][0] - t <= 0.1 else None
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            if t0 <= t <= t1:
                w = (t - t0) / max(t1 - t0, 1e-9)
                return p0 + w * (p1 - p0)
        return samples[-1][1] if t - samples[-1][0] <= 0.1 else None

    def _kink_single(self, kink: Kink) -> Optional[Event]:
        """A collision with a ball that did not turn a clean corner of its own."""
        d = self.table.ball_diameter_in
        lo, hi = 0.75 * d, self.cfg.events.kink_pair_distance_ball_diameters * d
        best, best_gap = None, None
        for other_id in self._observed:
            if other_id == kink.track_id:
                continue
            pos = self._position_at(other_id, kink.t)
            if pos is None:
                continue
            gap = float(np.linalg.norm(pos - kink.point))
            if not lo <= gap <= hi:
                continue
            # The kinked ball must have been heading into it.
            if float(np.dot(kink.v_in, pos - kink.point)) <= 0:
                continue
            if best_gap is None or abs(gap - d) < abs(best_gap - d):
                best, best_gap = (other_id, pos), gap
        if best is None:
            return None
        other_id, pos = best
        still = Kink(other_id, kink.t, pos, np.zeros(2), np.zeros(2))
        return self._kink_collision(kink, still)

    def _rail_reversal(
        self, track_id: int, history: Sequence[Tuple[float, np.ndarray]]
    ) -> Optional[Event]:
        """A cushion contact, as motion toward a rail turning into motion away.

        Real bounces are not clean corners.  The ball slows into the cushion
        over two or three frames, spin bends its path out of it, a collision
        often happens a few frames before, and the ball may not be seen at the
        moment of contact at all.  The corner fit in ``find_kink`` wants two
        straight lines and finds almost none of that on broadcast footage.

        What every bounce does have is its component of motion *normal to the
        rail* changing sign.  So for each rail, each raw step of the ball is
        classed as approaching or leaving it (steps slower than the noise
        floor leave the state alone), and a change from approaching to leaving
        is a contact if the ball got within a ball's reach of that rail: either
        its closest observed position, or -- when it was not seen at the
        bounce -- the point where its approach and departure, extended, meet.
        """
        cfg = self.cfg.events
        samples = list(history)
        if len(samples) < 2:
            return None
        (t0, p0), (t1, p1) = samples[-2], samples[-1]
        step_t = t1 - t0
        if step_t <= 1e-6:
            return None
        r = self.table.ball_radius_in
        reach_hi = r * (1.0 + cfg.cushion_contact_tolerance_ball_radii)
        reach_lo = r - 1.0
        noise = cfg.min_closing_speed_in_s
        states = self._rail_state.setdefault(track_id, {})
        found: Optional[Event] = None
        for rail, (axis, limit, inward) in enumerate(self._rails()):
            n0 = p0[axis] if inward > 0 else limit - p0[axis]
            n1 = p1[axis] if inward > 0 else limit - p1[axis]
            approach = (n0 - n1) / step_t  # > 0 moving toward this rail
            previous = states.get(rail)
            if approach >= noise:
                states[rail] = ("approaching", t1, n1, approach, p1)
                continue
            if approach > -noise or previous is None or previous[0] != "approaching":
                if approach <= -noise:
                    states[rail] = ("leaving", t1, n1, -approach, p1)
                elif (
                    previous is not None and previous[0] == "approaching"
                    and t1 - previous[1] > cfg.cushion_approach_max_age_s
                ):
                    # Seen standing, or all but, since it last closed on this
                    # rail: whatever moves it next is a new motion, not the
                    # far side of a bounce.  (A ball *unseen* since, in the
                    # jaws or under a hand, keeps its approach.)
                    del states[rail]
                continue
            # Approaching -> leaving: where was it closest?
            _, t_a, n_a, speed_in, p_a = previous
            closest = min(n_a, n0, n1)
            # If unseen at the bounce, meet the two lines in (t, distance).
            speed_out = -approach
            t_meet = (n_a - n1 + speed_in * t_a + speed_out * t1) / (speed_in + speed_out)
            n_meet = n_a - speed_in * (t_meet - t_a)
            states[rail] = ("leaving", t1, n1, speed_out, p1)
            near = reach_lo <= closest <= reach_hi or (
                t_a <= t_meet <= t1 and reach_lo <= n_meet <= reach_hi
            )
            if not near or speed_in < cfg.min_closing_speed_in_s:
                continue
            where = min(((n_a, p_a), (n0, p0), (n1, p1)), key=lambda c: c[0])[1]
            when = float(np.clip(t_meet, t_a, t1))
            if self._another_ball_near(track_id, where, when):
                # Turned round next to another ball: that ball did it, not the
                # rail.  The collision is the kink detector's to report.
                continue
            event = self._cushion_event(track_id, where, when, axis, speed_in, closest)
            if event is not None:
                found = event
        return found

    def _another_ball_near(self, track_id: int, point: np.ndarray, t: float) -> bool:
        reach = self.cfg.events.kink_pair_distance_ball_diameters * self.table.ball_diameter_in
        for other in self._observed:
            if other == track_id:
                continue
            pos = self._position_at(other, t)
            if pos is not None and float(np.linalg.norm(pos - point)) <= reach:
                return True
        return False

    def _rails(self) -> List[Tuple[int, float, float]]:
        """(axis, far limit, inward sign) for the four rails."""
        L, W = self.table.length_in, self.table.width_in
        return [(0, L, 1.0), (0, L, -1.0), (1, W, 1.0), (1, W, -1.0)]

    def _cushion_event(
        self, track_id: int, point: np.ndarray, t_scene: float, axis: int,
        speed_in_s: float, gap_in: float,
    ) -> Optional[Event]:
        cfg = self.cfg.events
        if self.table.has_pockets and self.table.nearest_pocket_distance(
            (float(point[0]), float(point[1]))
        ) < cfg.cushion_pocket_clearance_ball_diameters * self.table.ball_diameter_in:
            return None
        frame, t_s = self._frame_at(t_scene)
        last = self._last_cushion.get(track_id)
        if last is not None and abs(t_s - last) < cfg.refractory_s:
            return None
        self._last_cushion[track_id] = t_s
        img = self.table.ball_table_to_image([tuple(point)])[0]
        return Event(
            type=EventType.CUSHION,
            frame=frame,
            t_s=t_s,
            table_xy=(float(point[0]), float(point[1])),
            image_xy=(float(img[0]), float(img[1])),
            track_ids=(track_id,),
            detail={"axis": float(axis), "speed_in_s": speed_in_s,
                    "rail_distance_in": gap_in, "from_path": 1.0},
        )

    def _kink_cushion(self, kink: Kink) -> Optional[Event]:
        cfg = self.cfg.events
        r = self.table.ball_radius_in
        reach = r * (1.0 + cfg.cushion_contact_tolerance_ball_radii)
        min_normal = cfg.min_closing_speed_in_s
        p = kink.point
        if self.table.has_pockets and self.table.nearest_pocket_distance(
            (float(p[0]), float(p[1]))
        ) < cfg.cushion_pocket_clearance_ball_diameters * self.table.ball_diameter_in:
            # Rattling in the jaws is not a cushion contact worth reporting,
            # and a ball dropping into the pocket turns a corner too.
            return None

        best: Optional[Tuple[float, int, float]] = None
        for axis, limit in ((0, self.table.length_in), (1, self.table.width_in)):
            for inward, gap in ((1.0, float(p[axis])), (-1.0, limit - float(p[axis]))):
                toward = -inward * float(kink.v_in[axis])
                away = inward * float(kink.v_out[axis])
                if toward < min_normal or away < min_normal:
                    continue
                if not (r - 1.0 <= gap <= reach):
                    continue
                if best is None or abs(gap - r) < abs(best[0] - r):
                    best = (gap, axis, toward)
        if best is None:
            return None

        gap, axis, toward = best
        if self._another_ball_near(kink.track_id, p, kink.t):
            return None
        last = self._last_cushion.get(kink.track_id)
        frame, t_s = self._frame_at(kink.t)
        if last is not None and abs(t_s - last) < cfg.refractory_s:
            return None
        self._last_cushion[kink.track_id] = t_s
        img = self.table.ball_table_to_image([tuple(p)])[0]
        return Event(
            type=EventType.CUSHION,
            frame=frame,
            t_s=t_s,
            table_xy=(float(p[0]), float(p[1])),
            image_xy=(float(img[0]), float(img[1])),
            track_ids=(kink.track_id,),
            detail={
                "axis": float(axis),
                "speed_in_s": toward,
                "rail_distance_in": gap,
                "from_path": 1.0,
            },
        )

    def _kink_partner(self, kink: Kink) -> Optional[Kink]:
        """The pending kink of another ball that this one collided with, if any."""
        cfg = self.cfg.events
        reach = self.table.ball_diameter_in * cfg.kink_pair_distance_ball_diameters
        best, best_cost = None, None
        for other in self._pending:
            if other.track_id == kink.track_id:
                continue
            dt = abs(other.t - kink.t)
            gap = float(np.linalg.norm(other.point - kink.point))
            if dt > cfg.kink_pair_window_s or gap > reach:
                continue
            # They must have been closing on each other beforehand.
            direction = (kink.point - other.point) / max(gap, 1e-6)
            closing = float(np.dot(other.v_in - kink.v_in, direction))
            if closing < cfg.min_closing_speed_in_s:
                continue
            cost = dt / cfg.kink_pair_window_s + gap / reach
            if best_cost is None or cost < best_cost:
                best, best_cost = other, cost
        if best is not None:
            self._pending.remove(best)
        return best

    def _kink_collision(self, a: Kink, b: Kink) -> Optional[Event]:
        cfg = self.cfg.events
        t = min(a.t, b.t)
        frame, t_s = self._frame_at(t)
        key = frozenset((a.track_id, b.track_id))
        last = self._last_pair_event.get(key)
        if last is not None and abs(t_s - last) < cfg.refractory_s:
            return None
        self._last_pair_event[key] = t_s
        gap = float(np.linalg.norm(b.point - a.point))
        direction = (b.point - a.point) / max(gap, 1e-6)
        closing = float(np.dot(a.v_in - b.v_in, direction))
        mid = (a.point + b.point) / 2.0
        img = self.table.ball_table_to_image([tuple(mid)])[0]
        first, second = sorted((a, b), key=lambda k: k.t)
        return Event(
            type=EventType.COLLISION,
            frame=frame,
            t_s=t_s,
            table_xy=(float(mid[0]), float(mid[1])),
            image_xy=(float(img[0]), float(img[1])),
            track_ids=(first.track_id, second.track_id),
            detail={
                "separation_in": gap,
                "closing_speed_in_s": closing,
                "from_path": 1.0,
            },
        )

    # -- detectors ---------------------------------------------------------

    def _collisions(
        self, tracks: Sequence[Track], frame: int, t_s: float, dt: float
    ) -> List[Event]:
        """Contact, tested over the whole frame rather than at the end of it.

        Testing only where the balls are *now* cannot work at the speeds a
        break reaches.  Two balls are in contact over a shell 0.27 in thick --
        from 1.12 diameters apart down to touching -- and a cue ball crossing
        the table covers three or four inches between frames, so it is sampled
        inside that shell about one time in fifteen.  On ``fedor_shot.mp4`` the
        gap between the cue ball and the ball it pocketed read 3.72 in on one
        frame and 2.53 in on the next, against a 2.52 in threshold: the shot
        potted a ball and reported no collision.

        Both balls travel in a straight line over one frame, so their closest
        approach *during* the frame is exact arithmetic, and that is what the
        contact distance is compared against.  The closing speed is then how
        fast the gap actually shrank over the frame, which is what stops two
        balls resting against each other from emitting a collision forever:
        their gap is not shrinking.
        """
        cfg = self.cfg.events
        contact = cfg.contact_distance_ball_diameters * self.table.ball_diameter_in
        out: List[Event] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                pa, pb = a.kf.position, b.kf.position
                qa = self._prev_position.get(a.track_id, pa)
                qb = self._prev_position.get(b.track_id, pb)

                gap = qb - qa
                change = (pb - pa) - gap
                s, dist = closest_approach(gap, change)
                if dist > contact or dist < 1e-6:
                    continue

                closing = (float(np.linalg.norm(gap)) - dist) / dt
                if closing < cfg.min_closing_speed_in_s:
                    continue

                key = frozenset((a.track_id, b.track_id))
                last = self._last_pair_event.get(key)
                if last is not None and (t_s - last) < cfg.refractory_s:
                    continue
                self._last_pair_event[key] = t_s

                # Report the contact point, not the balls' current positions:
                # by now they have bounced apart.
                mid = (qa + s * (pa - qa) + qb + s * (pb - qb)) / 2.0
                img = self.table.ball_table_to_image([tuple(mid)])[0]
                event = Event(
                    type=EventType.COLLISION,
                    frame=frame,
                    t_s=t_s,
                    table_xy=(float(mid[0]), float(mid[1])),
                    image_xy=(float(img[0]), float(img[1])),
                    track_ids=(a.track_id, b.track_id),
                    detail={
                        "separation_in": dist,
                        "closing_speed_in_s": closing,
                    },
                )
                self.contacts_now.append(event.track_ids)
                still = [t for t, q in ((a, qa), (b, qb)) if self._was_at_rest(t)]
                if len(still) == 1:
                    q = qa if still[0] is a else qb
                    self._held.append((event, still[0].track_id, q.copy(), self._t_scene + _CONTACT_CONFIRM_S))
                else:
                    out.append(event)
        return out

    def _was_at_rest(self, track: Track) -> bool:
        v = self._prev_velocity.get(track.track_id)
        return v is not None and float(np.linalg.norm(v)) < self._rest_speed

    def _confirm_contacts(self, confirmed: Sequence[Track]) -> List[Event]:
        """Contacts with a ball at rest, once that ball has moved.

        A ball hit from rest moves.  One that sits where it was was not hit:
        the cue ball jumped over it -- ``albin_fedor``'s first shot, a jump
        over the 6 to pot the 4, was reported as "hit the 6 first" -- or
        passed it closer on the screen than on the table.  Held for
        ``_CONTACT_CONFIRM_S``, the contact is reported, at the frame it
        happened, once the ball has gone half a ball's width, or if by then
        it is out of sight (in the pocket, or behind the other); and dropped
        if it is still there.
        """
        if not self._held:
            return []
        by_id = {t.track_id: t for t in confirmed}
        limit = _CONTACT_MOVE_DIAMETERS * self.table.ball_diameter_in
        out: List[Event] = []
        keep = []
        for event, tid, was, deadline in self._held:
            track = by_id.get(tid)
            if track is not None and float(np.linalg.norm(track.kf.position - was)) >= limit:
                out.append(event)
            elif self._t_scene >= deadline:
                if track is None or track.time_since_update > 0:
                    out.append(event)
            else:
                keep.append((event, tid, was, deadline))
        self._held = keep
        return out

    def _picture_moved(self, struck: Sequence[Event], confirmed: Sequence[Track]) -> bool:
        """Whether balls "struck" together are the picture moving, not play.

        One stroke of the cue sets one ball moving, and it sets the others
        moving by hitting them.  So two balls at rest, far apart, that start
        moving on the same frame while nothing else is moving were not
        struck: the camera moved, or the broadcast is dissolving to another
        camera and the table fitted to the first no longer fits.  On the
        2026 Premier League final every one of its seven dissolves did that,
        three to eight balls at once, and opened a shot; one "potted" the 9.
        """
        if len(struck) < 2:
            return False
        ids = {tid for e in struck for tid in e.track_ids}
        if any(
            t.track_id not in ids and t.age >= _MIN_AGE_FOR_STRUCK
            and self._motion_state.get(t.track_id) == "moving"
            for t in confirmed
        ):
            return False
        pts = np.array([e.table_xy for e in struck], dtype=np.float64)
        spread = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2)))
        return spread > _TOGETHER_BALL_DIAMETERS * self.table.ball_diameter_in

    def _jump_speed(self, track: Track) -> float:
        """How fast the ball went between its last two sightings, if it was
        just seen more than a ball's width from the one before, within
        ``_JUMP_WINDOW_S``; else 0."""
        trail = track.trail
        if len(trail) < 2 or not trail[-1].observed:
            return 0.0
        now = trail[-1]
        for k in range(2, min(len(trail), _JUMP_MAX_SAMPLES + 1) + 1):
            before = trail[-k]
            if not before.observed:
                continue
            dt = now.t_s - before.t_s
            if dt <= 0 or dt > _JUMP_WINDOW_S:
                return 0.0
            step = float(np.hypot(now.table_xy[0] - before.table_xy[0], now.table_xy[1] - before.table_xy[1]))
            return step / dt if step >= self.table.ball_diameter_in else 0.0
        return 0.0

    def _balls_struck(
        self, tracks: Sequence[Track], frame: int, t_s: float
    ) -> List[Event]:
        """A ball going from rest to moving under its own steam.

        This needs a proper two-threshold state machine, and both thresholds
        exist for a reason:

        * A single **low** threshold for the remembered state means a ball
          accelerating through it sets "already moving" on the way up, and the
          strike is never reported -- a clip with five pots produced no shots.
        * A single threshold of *any* value means a ball whose estimated speed
          wobbles across it fires on every upward crossing; one ball emitted
          eight "struck" events in half a second.

        With a hysteresis band, a ball must genuinely come to rest (below
        ``_rest_speed``) before it can be struck again (above
        ``_struck_speed``), and anything in between leaves the state alone.
        """
        out: List[Event] = []
        struck_speed = self._struck_speed
        rest_speed = self._rest_speed

        for track in tracks:
            speed = track.speed
            state = self._motion_state.get(track.track_id)

            if state is None:
                # First sighting: adopt whatever it is doing, and say nothing.
                # A brand-new track also starts with a deliberately wide
                # velocity prior, so its first estimates are not trustworthy.
                self._motion_state[track.track_id] = (
                    "moving" if speed > struck_speed else "rest"
                )
                continue

            if state == "moving":
                if speed < rest_speed:
                    self._motion_state[track.track_id] = "rest"
                continue

            if speed <= struck_speed:
                # Struck and into another ball between two sightings, the
                # ball's filtered speed can stay under the threshold: on the
                # tripod answer key the cue ball jumped 69 px in two frames,
                # hidden by the cue for one, and read 22 in/s.
                speed = max(speed, self._jump_speed(track))
            if speed <= struck_speed or track.age < _MIN_AGE_FOR_STRUCK:
                continue
            if self._t_scene < self._quiet_until:
                # Back from a cut: moving already, not struck just now.
                self._motion_state[track.track_id] = "moving"
                continue
            self._motion_state[track.track_id] = "moving"
            pos = track.kf.position
            img = self.table.ball_table_to_image([tuple(pos)])[0]
            out.append(
                Event(
                    type=EventType.BALL_STRUCK,
                    frame=frame,
                    t_s=t_s,
                    table_xy=(float(pos[0]), float(pos[1])),
                    image_xy=(float(img[0]), float(img[1])),
                    track_ids=(track.track_id,),
                    detail={"speed_in_s": speed},
                )
            )
        return out

    # -- summary -----------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts
