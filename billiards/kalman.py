"""A 2-D constant-velocity Kalman filter with rolling friction.

Two decisions here matter more than the filter itself:

1.  **It runs in table inches, not image pixels.**  In the image, a ball rolling
    at constant speed accelerates as it comes towards the camera -- perspective
    makes uniform motion look non-uniform, so a constant-velocity model fitted
    in pixel space is permanently wrong.  After the homography, uniform motion
    really is uniform, and the model matches the physics.

2.  **Velocity decays exponentially** rather than staying constant.  A ball
    rolling on cloth loses speed continuously; modelling that means a track
    that is occluded for half a second is predicted to the right place instead
    of overshooting past it.

The state is ``[x, y, vx, vy]`` with position in inches and velocity in
inches/second.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


#: Normalised-innovation-squared above which a measurement is treated as
#: evidence of a manoeuvre rather than noise.  The NIS of a correct 2-D model is
#: chi-squared with 2 degrees of freedom, so 9 is roughly the 99th percentile --
#: it fires on real events, not on jitter.
_MANOEUVRE_NIS = 9.0

#: The adaptive gain below which the model counts as having predicted the
#: last measurement: a normalised innovation under about 4.5, against 2 for a
#: model that is exactly right.
_SETTLED_GAIN = 1.5


class BallKalman:
    def __init__(
        self,
        initial_position: Tuple[float, float],
        *,
        velocity_tau_s: float = 3.6,
        accel_std_in_s2: float = 60.0,
        meas_std_in: float = 0.22,
        init_vel_std_in_s: float = 120.0,
        initial_velocity: Tuple[float, float] = (0.0, 0.0),
        manoeuvre_gain_max: float = 25.0,
    ) -> None:
        self.tau = float(max(velocity_tau_s, 1e-3))
        self.accel_std = float(accel_std_in_s2)
        self.meas_std = float(meas_std_in)
        self.manoeuvre_gain_max = float(manoeuvre_gain_max)
        #: Adaptive process-noise multiplier, driven by how surprising the last
        #: measurement was.  See ``update``.
        self._q_gain = 1.0

        self.x = np.array(
            [initial_position[0], initial_position[1],
             initial_velocity[0], initial_velocity[1]],
            dtype=np.float64,
        )
        self.P = np.diag(
            [
                meas_std_in**2,
                meas_std_in**2,
                init_vel_std_in_s**2,
                init_vel_std_in_s**2,
            ]
        ).astype(np.float64)

        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * (meas_std_in**2)

    # -- accessors ---------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:]))

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[:2, :2].copy()

    @property
    def settled(self) -> bool:
        """Did the model see the last measurement coming?

        False for a frame or two after a surprise -- a collision, a cushion, a
        ball just struck -- while the velocity estimate is still catching up
        with what the ball is actually doing.
        """
        return self._q_gain < _SETTLED_GAIN

    # -- model -------------------------------------------------------------

    def _transition(self, dt: float) -> np.ndarray:
        """Exact solution of dv/dt = -v/tau over ``dt``.

        Position advances by ``v * tau * (1 - e^{-dt/tau})``, which tends to
        ``v * dt`` as tau grows -- i.e. it degrades gracefully to the textbook
        constant-velocity model when friction is switched off.
        """
        decay = float(np.exp(-dt / self.tau))
        travel = self.tau * (1.0 - decay)
        return np.array(
            [
                [1.0, 0.0, travel, 0.0],
                [0.0, 1.0, 0.0, travel],
                [0.0, 0.0, decay, 0.0],
                [0.0, 0.0, 0.0, decay],
            ],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        """Continuous white-noise-acceleration covariance, adaptively scaled.

        There are two conflicting requirements here, and a single fixed
        ``accel_std`` cannot meet both:

        * A **freely rolling** ball accelerates only through friction, a few
          in/s^2.  Keeping the process noise that low is what stops a ball that
          is sitting still from accumulating phantom velocity out of detection
          noise -- with a large fixed value, a stationary ball reads as moving
          at 15-20 in/s and every speed-based rule downstream breaks.
        * A **collision or cushion bounce** reverses the velocity between one
          frame and the next.  Following that needs process noise orders of
          magnitude larger, or the filter smooths the impact away and drags the
          track straight through the cushion.

        So the base value is the physical one, and ``update`` raises the gain
        for a few frames whenever a measurement arrives that the model did not
        see coming.  The filter is quiet when the table is quiet and loose
        exactly when something happens.
        """
        q = (self.accel_std * self._q_gain) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    # -- filter ------------------------------------------------------------

    def predict(self, dt: float) -> np.ndarray:
        dt = float(max(dt, 1e-6))
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        # Relax back towards the quiet model; a manoeuvre is brief.
        self._q_gain = max(1.0, self._q_gain * 0.55)
        return self.position

    def peek(self, dt: float) -> np.ndarray:
        """Predicted position without mutating the filter."""
        dt = float(max(dt, 1e-6))
        return (self._transition(dt) @ self.x)[:2]

    def rescale_time(self, factor: float) -> None:
        """Re-express the velocity in a clock running ``factor`` times faster.

        Used when the scene clock's rate estimate changes: the velocity was
        learned from intervals timed at the old rate, so it is off by exactly
        their ratio, and the filter would otherwise take several frames to
        notice -- frames in which every step reads long.
        """
        f = float(factor)
        scale = np.array([1.0, 1.0, f, f])
        self.x = self.x * scale
        self.P = self.P * np.outer(scale, scale)

    def peek_many(self, dts: Sequence[float]) -> np.ndarray:
        """``peek`` for several intervals at once: an (n, 2) array."""
        dt = np.maximum(np.asarray(dts, dtype=np.float64), 1e-6)
        travel = self.tau * (1.0 - np.exp(-dt / self.tau))
        return self.x[None, :2] + travel[:, None] * self.x[None, 2:4]

    def update(self, measurement: Tuple[float, float]) -> None:
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Normalised innovation squared: how many standard deviations away the
        # measurement landed.  Around 2 (the measurement dimension) when the
        # model is right; large when the ball just did something the model
        # cannot represent -- which at a pool table means a collision or a
        # cushion, the two moments the whole tool exists to capture.
        try:
            nis = float(y @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically rare
            nis = 2.0

        if nis > _MANOEUVRE_NIS:
            # React in *this* frame rather than the next one.  Inflating the
            # covariance before computing the gain tells the filter "your
            # velocity estimate is stale, trust this measurement", so the track
            # turns the corner with the ball instead of one frame behind it.
            # Reacting a frame late is what leaves a visible overshoot through
            # every cushion and contact point.
            scale = float(
                np.clip(nis / _MANOEUVRE_NIS, 1.0, self.manoeuvre_gain_max**2)
            )
            self.P[2:, 2:] *= scale
            self.P[:2, :2] *= min(scale, 4.0)
            S = self.H @ self.P @ self.H.T + self.R

        self._q_gain = float(
            np.clip(np.sqrt(max(nis, 2.0) / 2.0), 1.0, self.manoeuvre_gain_max)
        )

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        # Joseph form: stays positive-definite even with an aggressive gain.
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def innovation_distance(self, measurement: Tuple[float, float]) -> float:
        """Mahalanobis distance of a measurement from the prediction."""
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            return float(np.sqrt(y @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:  # pragma: no cover - numerically rare
            return float(np.linalg.norm(y) / max(self.meas_std, 1e-6))

    def apply_impulse(self, velocity_std_in_s: float = 150.0) -> None:
        """Inflate velocity uncertainty after a detected collision.

        At a collision the velocity changes discontinuously.  Telling the filter
        "you no longer know the velocity" here is what stops it from smoothing
        the impact away and drawing the ball through its own contact point.

        The argument is a velocity standard deviation in in/s, matching the
        units of the velocity block of P.  (An earlier version added an
        *acceleration* variance here, which is dimensionally wrong and made the
        inflation scale with an unrelated quantity.)
        """
        self.P[2:, 2:] += np.eye(2) * float(velocity_std_in_s) ** 2
        self._q_gain = self.manoeuvre_gain_max
