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

from .assignment import FORBIDDEN, associate
from .config import Config
from .detect import ColorSignature, Detection, colour_distance_matrix
from .geometry import TableModel
from .kalman import BallKalman


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
        "colour_locked", "last_observed_xy",
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
        self.state = TrackState.TENTATIVE
        self.hits = 1
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
        self.colour_locked = False

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
        return self.signature.classify()

    @property
    def label(self) -> str:
        t = self.ball_type
        if t == "cue":
            return "CUE"
        if t == "eight":
            return "8"
        return f"#{self.track_id}"

    # -- filter interaction ------------------------------------------------

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, cfg: Config) -> None:
        self.kf.update(detection.centre_table)
        self.hits += 1
        self.time_since_update = 0
        self.last_image_xy = detection.centre_image
        self.last_observed_xy = detection.centre_table
        self.last_radius_px = detection.radius_px

        # Colour is averaged in slowly.  A detection taken mid-collision, or
        # one clipped by the cue stick, has a contaminated colour; a fast EMA
        # would let that poison the identity the tracker relies on.
        alpha = 0.35 if self.hits <= 3 else 0.08
        if not detection.from_cluster:
            self.signature = self.signature.blend(detection.signature, alpha)

        if self.state is TrackState.TENTATIVE:
            if self.hits >= cfg.tracker.min_hits_to_confirm:
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
        img = table.table_to_image([tuple(pos)])[0]
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
        trail_seconds = cfg.render.trail_seconds if cfg.render.trail_seconds > 0 else 8.0
        self._trail_cap = int(max(8, round(trail_seconds * self.fps)))
        self.last_stats: Dict[str, int] = {}

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

        for di in unmatched_dets:
            self._spawn(detections[di], frame)

        self._retire(frame)

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
            elif track.time_since_update > tc.max_age_coasting:
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
                self.finished.append(track)
            else:
                alive.append(track)
        self.tracks = alive

    # -- accessors ---------------------------------------------------------

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.is_visible]

    def all_tracks(self) -> List[Track]:
        return list(self.tracks) + list(self.finished)

    def cue_ball(self) -> Optional[Track]:
        """The track that currently looks most like the cue ball."""
        candidates = [t for t in self.active_tracks() if t.ball_type == "cue"]
        if not candidates:
            return None
        return max(candidates, key=lambda t: (t.signature.white_fraction, t.hits))

    def reset_trails(self) -> None:
        for track in self.tracks:
            track.trail.clear()
