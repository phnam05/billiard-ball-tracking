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

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .geometry import TableModel
from .track import Track, TrackState


#: Frames a track must have existed before its speed is trusted enough to say
#: it was struck.  The filter starts with a wide velocity prior by design.
_MIN_AGE_FOR_STRUCK = 8


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
        confirmed = [t for t in tracks if t.state is TrackState.CONFIRMED]
        new: List[Event] = []
        new.extend(self._collisions(confirmed, frame, t_s, dt))
        new.extend(self._cushions(confirmed, frame, t_s, dt))
        new.extend(self._balls_struck(confirmed, frame, t_s))

        for track in confirmed:
            self._prev_velocity[track.track_id] = track.velocity.copy()
            self._prev_position[track.track_id] = track.kf.position.copy()

        self.events.extend(new)
        return new

    def note_pot(self, track: Track, frame: int, t_s: float) -> Event:
        pos = track.kf.position
        img = self.table.table_to_image([tuple(pos)])[0]
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
                img = self.table.table_to_image([tuple(mid)])[0]
                out.append(
                    Event(
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
                )
        return out

    def _cushions(
        self, tracks: Sequence[Track], frame: int, t_s: float, dt: float
    ) -> List[Event]:
        cfg = self.cfg.events
        base_proximity = cfg.cushion_proximity_ball_radii * self.table.ball_radius_in
        out: List[Event] = []

        for track in tracks:
            prev = self._prev_velocity.get(track.track_id)
            if prev is None:
                continue
            vel = track.velocity
            if float(np.linalg.norm(vel - prev)) < cfg.min_bounce_speed_change_in_s:
                continue

            # How near the rail counts as "at" it has to include the distance
            # the ball covered since the previous frame.  A break travels ~200
            # in/s, which is nearly 7 inches per frame at 30 fps, so by the time
            # the reversal is observable the ball is already well off the
            # cushion.  A fixed 1.4-inch gate therefore matched essentially
            # never -- clips full of obvious bounces reported zero.
            proximity = base_proximity + track.speed * dt

            pos = track.kf.position
            x, y = float(pos[0]), float(pos[1])
            # Which cushion is nearest, and along which axis would it reflect?
            distances = {
                0: min(x, self.table.length_in - x),
                1: min(y, self.table.width_in - y),
            }
            axis = min(distances, key=lambda k: distances[k])
            if distances[axis] > proximity:
                continue
            # A genuine bounce reverses the velocity component normal to the rail.
            if prev[axis] * vel[axis] >= 0:
                continue

            last = self._last_cushion.get(track.track_id)
            if last is not None and (t_s - last) < cfg.refractory_s:
                continue
            self._last_cushion[track.track_id] = t_s

            img = self.table.table_to_image([(x, y)])[0]
            out.append(
                Event(
                    type=EventType.CUSHION,
                    frame=frame,
                    t_s=t_s,
                    table_xy=(x, y),
                    image_xy=(float(img[0]), float(img[1])),
                    track_ids=(track.track_id,),
                    detail={
                        "axis": float(axis),
                        "speed_in_s": track.speed,
                        "rail_distance_in": distances[axis],
                    },
                )
            )
        return out

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

            if speed <= struck_speed or track.age < _MIN_AGE_FOR_STRUCK:
                continue
            self._motion_state[track.track_id] = "moving"
            pos = track.kf.position
            img = self.table.table_to_image([tuple(pos)])[0]
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
