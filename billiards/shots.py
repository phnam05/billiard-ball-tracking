"""Group the raw event stream into shots.

A flat list of "collision at t=4.12s between track 3 and track 7" is accurate
but not readable.  What a pool player wants is a shot: *the cue ball was struck,
it hit the 4 first, the 4 went two cushions and dropped in the corner, and the
cue ball finished here*.

A shot is delimited by motion, not by a timer: it opens when a ball starts
moving from rest and closes once every ball has been below the stationary speed
for a short while.  That is the same definition a referee uses, and it needs no
threshold that is not already in the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config
from .events import Event, EventType
from .track import Track, TrackState


@dataclass
class Shot:
    index: int
    start_frame: int
    start_t_s: float
    end_frame: Optional[int] = None
    end_t_s: Optional[float] = None

    #: Track id of the ball that set off first -- normally the cue ball.
    opener_track_id: Optional[int] = None
    opener_label: str = "?"

    #: The first ball the opener touched, which is what "hit the 4 first" means.
    first_contact_track_id: Optional[int] = None
    first_contact_label: Optional[str] = None

    collisions: int = 0
    cushions: int = 0
    potted: List[str] = field(default_factory=list)
    #: Distance each involved ball travelled, in inches.
    travel_in: Dict[int, float] = field(default_factory=dict)
    peak_speed_in_s: float = 0.0

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_t_s is None:
            return None
        return self.end_t_s - self.start_t_s

    def describe(self) -> str:
        """One human-readable line."""
        parts = [f"shot {self.index}: {self.opener_label} struck"]
        if self.first_contact_label:
            parts.append(f"hit {self.first_contact_label} first")
        if self.cushions:
            parts.append(f"{self.cushions} cushion{'s' if self.cushions != 1 else ''}")
        if self.collisions > 1:
            parts.append(f"{self.collisions} contacts")
        if self.potted:
            parts.append("potted " + ", ".join(self.potted))
        else:
            parts.append("nothing potted")
        if self.duration_s is not None:
            parts.append(f"{self.duration_s:.1f}s")
        return " -- ".join(parts)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "start_t_s": round(self.start_t_s, 3),
            "end_frame": self.end_frame,
            "end_t_s": None if self.end_t_s is None else round(self.end_t_s, 3),
            "duration_s": None if self.duration_s is None else round(self.duration_s, 3),
            "opener": self.opener_label,
            "opener_track_id": self.opener_track_id,
            "first_contact": self.first_contact_label,
            "first_contact_track_id": self.first_contact_track_id,
            "collisions": self.collisions,
            "cushions": self.cushions,
            "potted": list(self.potted),
            "peak_speed_in_s": round(self.peak_speed_in_s, 1),
            "total_travel_in": round(sum(self.travel_in.values()), 1),
            "summary": self.describe(),
        }


class ShotSegmenter:
    """Consumes each frame's tracks and events, and emits completed shots."""

    def __init__(self, cfg: Config, fps: float) -> None:
        self.cfg = cfg
        self.fps = max(float(fps), 1.0)
        self.shots: List[Shot] = []
        self._current: Optional[Shot] = None
        self._still_frames = 0
        self._last_pos: Dict[int, np.ndarray] = {}
        #: Balls must be at rest this long before a shot is considered over,
        #: so a ball creeping to a stop does not end it early.
        self._settle_frames = max(3, int(round(0.4 * self.fps)))

    # -- per frame ---------------------------------------------------------

    def step(
        self, tracks: Sequence[Track], events: Sequence[Event], frame: int, t_s: float
    ) -> Optional[Shot]:
        """Returns a shot if one completed on this frame."""
        confirmed = [t for t in tracks if t.state is TrackState.CONFIRMED]
        threshold = self.cfg.tracker.stationary_speed_in_s
        moving = [t for t in confirmed if t.speed > threshold]

        completed: Optional[Shot] = None

        if self._current is None:
            starters = [e for e in events if e.type is EventType.BALL_STRUCK]
            if starters:
                opener = self._pick_opener(starters, confirmed)
                self._current = Shot(
                    index=len(self.shots) + 1,
                    start_frame=frame,
                    start_t_s=t_s,
                    opener_track_id=None if opener is None else opener.track_id,
                    opener_label="?" if opener is None else opener.label,
                )
                self._still_frames = 0
                self._last_pos = {t.track_id: t.kf.position.copy() for t in confirmed}
        else:
            self._accumulate(confirmed, events)

            if moving:
                self._still_frames = 0
            else:
                self._still_frames += 1
                if self._still_frames >= self._settle_frames:
                    self._current.end_frame = frame
                    self._current.end_t_s = t_s
                    self.shots.append(self._current)
                    completed = self._current
                    self._current = None
                    self._last_pos.clear()

        return completed

    @staticmethod
    def _pick_opener(
        starters: Sequence[Event], confirmed: Sequence[Track]
    ) -> Optional[Track]:
        """Which ball opened the shot, when several started in the same frame.

        At 30 fps the cue ball and the ball it hits can cross the speed
        threshold on the same frame, and track order is arbitrary, so taking the
        first one reports the object ball as having been struck.  The cue ball
        is the right answer whenever one is on the table; failing that, the
        fastest ball is the one that was hit.
        """
        ids = {tid for e in starters for tid in e.track_ids}
        candidates = [t for t in confirmed if t.track_id in ids]
        if not candidates:
            return None
        cue = [t for t in candidates if t.ball_type == "cue"]
        if cue:
            return max(cue, key=lambda t: t.speed)
        return max(candidates, key=lambda t: t.speed)

    def _accumulate(self, confirmed: Sequence[Track], events: Sequence[Event]) -> None:
        shot = self._current
        assert shot is not None

        for track in confirmed:
            pos = track.kf.position
            previous = self._last_pos.get(track.track_id)
            if previous is not None:
                shot.travel_in[track.track_id] = shot.travel_in.get(
                    track.track_id, 0.0
                ) + float(np.linalg.norm(pos - previous))
            self._last_pos[track.track_id] = pos.copy()
            shot.peak_speed_in_s = max(shot.peak_speed_in_s, track.speed)

        by_id = {t.track_id: t for t in confirmed}
        for e in events:
            if e.type is EventType.COLLISION:
                shot.collisions += 1
                if shot.first_contact_track_id is None and shot.opener_track_id in e.track_ids:
                    other = next(
                        (i for i in e.track_ids if i != shot.opener_track_id), None
                    )
                    if other is not None:
                        shot.first_contact_track_id = other
                        target = by_id.get(other)
                        shot.first_contact_label = (
                            target.label if target is not None else f"#{other}"
                        )
            elif e.type is EventType.CUSHION:
                shot.cushions += 1
            elif e.type is EventType.POT:
                for tid in e.track_ids:
                    target = by_id.get(tid)
                    shot.potted.append(
                        target.label if target is not None else f"#{tid}"
                    )

    # -- results -----------------------------------------------------------

    def finish(self, frame: int, t_s: float) -> None:
        """Close an in-progress shot at the end of the clip."""
        if self._current is not None:
            self._current.end_frame = frame
            self._current.end_t_s = t_s
            self.shots.append(self._current)
            self._current = None

    def to_list(self) -> List[dict]:
        return [s.to_dict() for s in self.shots]

    def summary_lines(self) -> List[str]:
        return [s.describe() for s in self.shots]
