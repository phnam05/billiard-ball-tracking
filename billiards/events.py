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
    def __init__(self, cfg: Config, table: TableModel) -> None:
        self.cfg = cfg
        self.table = table
        self.events: List[Event] = []
        self._last_pair_event: Dict[frozenset, float] = {}
        self._last_cushion: Dict[int, float] = {}
        self._prev_velocity: Dict[int, np.ndarray] = {}
        self._was_moving: Dict[int, bool] = {}

    @property
    def _shot_speed(self) -> float:
        """Speed at which a ball counts as having been struck.

        The *same* number has to serve both as the event threshold and as the
        state remembered between frames.  Using a lower one for the state (the
        stationary cut-off) let a ball accelerating through the gap set
        "already moving" on its way up, so the strike itself was never reported
        -- which is why a clip containing five pots produced zero shots.
        """
        return max(self.cfg.tracker.stationary_speed_in_s * 6.0, 12.0)

    # -- public ------------------------------------------------------------

    def step(
        self, tracks: Sequence[Track], frame: int, t_s: float
    ) -> List[Event]:
        confirmed = [t for t in tracks if t.state is TrackState.CONFIRMED]
        new: List[Event] = []
        new.extend(self._collisions(confirmed, frame, t_s))
        new.extend(self._cushions(confirmed, frame, t_s))
        new.extend(self._balls_struck(confirmed, frame, t_s))

        for track in confirmed:
            self._prev_velocity[track.track_id] = track.velocity.copy()
            self._was_moving[track.track_id] = track.speed > self._shot_speed

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
        self, tracks: Sequence[Track], frame: int, t_s: float
    ) -> List[Event]:
        cfg = self.cfg.events
        contact = cfg.contact_distance_ball_diameters * self.table.ball_diameter_in
        out: List[Event] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                pa, pb = a.kf.position, b.kf.position
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                if dist > contact or dist < 1e-6:
                    continue

                direction = delta / dist
                closing = float(np.dot(a.velocity - b.velocity, direction))
                if closing < cfg.min_closing_speed_in_s:
                    continue

                key = frozenset((a.track_id, b.track_id))
                last = self._last_pair_event.get(key)
                if last is not None and (t_s - last) < cfg.refractory_s:
                    continue
                self._last_pair_event[key] = t_s

                mid = (pa + pb) / 2.0
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
        self, tracks: Sequence[Track], frame: int, t_s: float
    ) -> List[Event]:
        cfg = self.cfg.events
        proximity = cfg.cushion_proximity_ball_radii * self.table.ball_radius_in
        out: List[Event] = []

        for track in tracks:
            prev = self._prev_velocity.get(track.track_id)
            if prev is None:
                continue
            vel = track.velocity
            if float(np.linalg.norm(vel - prev)) < cfg.min_bounce_speed_change_in_s:
                continue

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
        out: List[Event] = []
        for track in tracks:
            was = self._was_moving.get(track.track_id)
            if was is None:
                continue
            if was or track.speed <= self._shot_speed:
                continue
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
                    detail={"speed_in_s": track.speed},
                )
            )
        return out

    # -- summary -----------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts
