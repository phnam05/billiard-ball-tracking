#!/usr/bin/env python
"""Generate a synthetic pool clip *with ground truth*.

Why this exists: "the tracking was not very robust" is impossible to fix
reliably if the only way to judge a change is to watch a video and squint.  This
renders a physically simulated rack from a virtual camera at a realistic angle,
and writes the exact table-space position of every ball in every frame.  That
turns tracking quality into a number -- median position error in inches, and
whether ball identities survive the shot -- which the test suite then asserts on.

    python tools/make_synthetic_clip.py --out assets/synthetic_break.mp4

The simulation is deliberately harsh: balls start racked (so the detector must
split a dense cluster), they collide repeatedly, they bounce off cushions, and
the render adds perspective, a lighting gradient, sensor noise and JPEG-ish
softening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TABLE_L = 100.0  # inches, playing surface
TABLE_W = 50.0
BALL_R = 1.125

#: Ball colours in BGR, as a broadcast camera records them: measured off the
#: three sample clips (the median colour of each ball's coloured pixels, see
#: DIARY.md 23 Sep), except orange, which no sample clip has: it is the
#: orange of ``billiards.balls.FAMILIES``, an estimate between their red and
#: yellow.  An earlier version used saturated guesses
#: whose hues were up to 49 degrees from any real ball's, so a palette that
#: named the synthetic balls correctly could not have named real ones.
#: The white of a ball -- the cue ball, a standard stripe's caps, every number
#: circle.  Ivory, as measured on the sample clips' cue ball (b* +9 on all
#: three), not a bluish screen white: on the sample clips' blue-grey cloth a
#: cool white falls inside the cloth's colour window, and a warm one does not.
IVORY: Tuple[int, int, int] = (166, 184, 190)

BALL_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "cue": IVORY,
    "yellow": (78, 140, 194),
    "blue": (99, 63, 26),
    "red": (85, 44, 162),
    "pink": (127, 105, 189),
    "purple": (101, 75, 84),
    "orange": (61, 93, 186),
    "green": (108, 129, 43),
    "maroon": (62, 68, 120),
    "black": (24, 24, 26),
}

#: Cloth colours, BGR.  ``broadcast`` is the blue-grey cloth of the sample
#: clips (HSV 109/26/176, the same on all three); ``green`` is the cloth the
#: simulator used until 23 Sep 2026.  The rest are other cloths a table is
#: covered in, for ``tools/robustness.py``: tournament blue, burgundy, camel,
#: and a nearly neutral grey, the hardest for a tracker that looks for "not
#: cloth" by colour.
CLOTHS: Dict[str, Tuple[int, int, int]] = {
    "broadcast": (176, 161, 147),
    "green": (86, 122, 46),
    "blue": (168, 92, 12),
    "red": (40, 28, 128),
    "tan": (104, 150, 184),
    "grey": (128, 124, 118),
}

#: The racks.  Each entry is (colour, number, striped).  ``standard`` is a
#: standard ball set (4 purple, 5 orange, stripes with white caps);
#: ``tv`` is the set in the sample broadcasts -- 4 pink, 5 purple, and stripes
#: whose caps are black -- racked for 9-ball, so the yellow 1 and the yellow
#: 9 are on the table together and only the stripe tells them apart.
RACKS: Dict[str, List[Tuple[str, int, bool]]] = {
    "standard": [
        ("yellow", 1, False), ("blue", 10, True), ("red", 3, False),
        ("purple", 4, False), ("orange", 13, True), ("green", 6, False),
        ("maroon", 15, True), ("black", 8, False),
    ],
    "tv": [
        ("yellow", 1, False), ("blue", 2, False), ("red", 3, False),
        ("pink", 4, False), ("yellow", 9, True), ("purple", 5, False),
        ("green", 6, False), ("maroon", 7, False), ("black", 8, False),
    ],
}
#: Colour of a stripe's caps, per ball set.
CAPS: Dict[str, Tuple[int, int, int]] = {
    "standard": IVORY,
    "tv": (30, 30, 32),
}


@dataclass
class Ball:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    colour: Tuple[int, int, int]
    striped: bool = False
    active: bool = True
    #: The number printed on it; 0 for the cue ball.
    number: int = 0
    caps: Tuple[int, int, int] = IVORY
    #: How far it has rolled, inches, which turns its number circle, and the
    #: table direction it last rolled in.
    rolled_in: float = 0.0
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------


class Simulation:
    def __init__(
        self,
        balls: Sequence[Ball],
        *,
        tau_s: float = 3.2,
        restitution: float = 0.92,
        pocket_radius: float = 2.4,
        seed: int = 0,
    ) -> None:
        self.balls = list(balls)
        self.tau = tau_s
        self.restitution = restitution
        self.pocket_radius = pocket_radius
        self.rng = np.random.default_rng(seed)
        self.pockets = np.array(
            [
                [0.0, 0.0], [TABLE_L / 2, 0.0], [TABLE_L, 0.0],
                [0.0, TABLE_W], [TABLE_L / 2, TABLE_W], [TABLE_L, TABLE_W],
            ]
        )
        #: Simulated time, and every contact the physics resolved.  This is the
        #: event ground truth: the tracker's collisions, cushions and pots are
        #: scored against it, the same way its positions are scored against the
        #: per-frame CSV.
        self.t = 0.0
        self.events: List[Dict[str, object]] = []

    def step(self, dt: float, substeps: int = 8) -> None:
        # Where every ball was when the frame began, so a collision between two
        # balls that were already touching -- momentum passing through a rack
        # -- can be told from one that a camera could actually see happen.
        self._frame_start = {id(b): b.pos.copy() for b in self.balls}
        h = dt / substeps
        for _ in range(substeps):
            self._substep(h)
            self.t += h

    def _log(self, kind: str, pos: np.ndarray, balls: Sequence[str], **detail: float) -> None:
        self.events.append({
            "type": kind,
            "t_s": round(self.t, 5),
            "x_in": round(float(pos[0]), 3),
            "y_in": round(float(pos[1]), 3),
            "balls": list(balls),
            **{k: round(float(v), 3) for k, v in detail.items()},
        })

    def _substep(self, h: float) -> None:
        decay = math.exp(-h / self.tau)
        for b in self.balls:
            if not b.active:
                continue
            step = b.vel * self.tau * (1.0 - decay)
            b.pos = b.pos + step
            travelled = float(np.linalg.norm(step))
            if travelled > 1e-9:
                b.rolled_in += travelled
                b.heading = step / travelled
            b.vel = b.vel * decay
            if np.linalg.norm(b.vel) < 0.35:
                b.vel[:] = 0.0

        self._cushions()
        self._collisions()
        self._pockets()

    def _cushions(self) -> None:
        for b in self.balls:
            if not b.active:
                continue
            for axis, limit in ((0, TABLE_L), (1, TABLE_W)):
                if b.pos[axis] < BALL_R:
                    b.pos[axis] = BALL_R + (BALL_R - b.pos[axis])
                    self._log("cushion", b.pos, [b.name], axis=axis,
                              speed_in_s=abs(b.vel[axis]))
                    b.vel[axis] = -b.vel[axis] * self.restitution
                elif b.pos[axis] > limit - BALL_R:
                    b.pos[axis] = (limit - BALL_R) - (b.pos[axis] - (limit - BALL_R))
                    self._log("cushion", b.pos, [b.name], axis=axis,
                              speed_in_s=abs(b.vel[axis]))
                    b.vel[axis] = -b.vel[axis] * self.restitution

    def _collisions(self) -> None:
        active = [b for b in self.balls if b.active]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                delta = b.pos - a.pos
                dist = float(np.linalg.norm(delta))
                if dist >= 2 * BALL_R or dist < 1e-9:
                    continue
                n = delta / dist
                # Separate so they are exactly touching.
                overlap = 2 * BALL_R - dist
                a.pos = a.pos - n * (overlap / 2)
                b.pos = b.pos + n * (overlap / 2)
                # Equal-mass elastic exchange of the normal components.
                va_n = float(np.dot(a.vel, n))
                vb_n = float(np.dot(b.vel, n))
                if va_n - vb_n <= 0:
                    continue
                start = getattr(self, "_frame_start", {})
                gap0 = (
                    float(np.linalg.norm(start[id(a)] - start[id(b)])) - 2 * BALL_R
                    if id(a) in start and id(b) in start else np.inf
                )
                self._log("collision", (a.pos + b.pos) / 2.0, [a.name, b.name],
                          closing_speed_in_s=va_n - vb_n,
                          touching_before=float(gap0 < 0.1))
                e = self.restitution
                a.vel = a.vel + n * (-(1 + e) / 2 * (va_n - vb_n))
                b.vel = b.vel + n * ((1 + e) / 2 * (va_n - vb_n))

    def _pockets(self) -> None:
        for b in self.balls:
            if not b.active:
                continue
            d = np.linalg.norm(self.pockets - b.pos, axis=1)
            if float(d.min()) < self.pocket_radius:
                b.active = False
                self._log("pot", b.pos, [b.name])

    @property
    def moving(self) -> bool:
        return any(
            b.active and float(np.linalg.norm(b.vel)) > 0.5 for b in self.balls
        )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


#: Pinhole cameras, in table inches (x along the length, y across, z up).
#: ``end`` is the camera recovered from the three sample broadcasts --
#: behind an end rail, 135 in back and 68 in up, a 36-degree lens -- so the
#: synthetic clip is filmed the way the real ones were.  ``side`` looks
#: across the table from in front of a long rail.
CAMERAS: Dict[str, Dict[str, object]] = {
    "end": {"position": (TABLE_L + 135.0, TABLE_W / 2 + 2.0, 68.0),
            "target": (TABLE_L * 0.47, TABLE_W / 2, 0.0), "fov_deg": 36.5},
    "side": {"position": (TABLE_L / 2 + 5.0, TABLE_W + 95.0, 80.0),
             "target": (TABLE_L / 2, TABLE_W * 0.45, 0.0), "fov_deg": 62.0},
    # A camera on the ceiling (or a phone on a boom) looking straight down;
    # "up" says which way the top of the picture faces, since straight down
    # has no horizon to take it from.
    "overhead": {"position": (TABLE_L / 2, TABLE_W / 2, 150.0),
                 "target": (TABLE_L / 2, TABLE_W / 2, 0.0), "fov_deg": 44.0,
                 "up": (0.0, 1.0, 0.0)},
    # A tripod at a corner of the room, looking across the table diagonally.
    "corner": {"position": (TABLE_L + 70.0, TABLE_W + 60.0, 95.0),
               "target": (TABLE_L * 0.45, TABLE_W * 0.45, 0.0), "fov_deg": 55.0},
}


class Renderer:
    """Render the table overhead, then warp it through a pinhole camera."""

    def __init__(
        self,
        out_size: Tuple[int, int] = (1280, 720),
        px_per_inch: float = 11.0,
        cloth_bgr: Tuple[int, int, int] = CLOTHS["broadcast"],
        camera: str = "end",
        seed: int = 0,
        parallax: bool = True,
    ) -> None:
        self.out_size = out_size
        self.ppi = px_per_inch
        self.cloth = cloth_bgr
        self.rng = np.random.default_rng(seed)

        self.rail_in = 5.0
        self.canvas_w = int((TABLE_L + 2 * self.rail_in) * px_per_inch)
        self.canvas_h = int((TABLE_W + 2 * self.rail_in) * px_per_inch)
        self.K, self.R, self.t = self._pinhole(CAMERAS[camera])
        #: table inches -> image, for the cloth and for the plane through the
        #: balls' centres, which is a ball radius above it.
        self._ground_H = self._plane_homography(0.0)
        self._ball_H = self._plane_homography(BALL_R if parallax else 0.0)
        to_table = np.array(
            [[1.0 / self.ppi, 0.0, -self.rail_in], [0.0, 1.0 / self.ppi, -self.rail_in],
             [0.0, 0.0, 1.0]]
        )
        self.H = self._ground_H @ to_table  # canvas pixels -> image
        self._background = self._make_background()
        self._venue = self._make_venue()
        self._vignette = self._make_vignette()

    # -- geometry ----------------------------------------------------------

    def _pinhole(self, spec: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = self.out_size
        f = (w / 2.0) / math.tan(math.radians(float(spec["fov_deg"])) / 2.0)
        K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
        C = np.asarray(spec["position"], dtype=np.float64)
        forward = np.asarray(spec["target"], dtype=np.float64) - C
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, spec.get("up", (0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        return K, R, -R @ C

    def _plane_homography(self, z: float) -> np.ndarray:
        """Table inches -> image pixels, for the horizontal plane at height ``z``."""
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t + z * self.R[:, 2]])

    def table_to_canvas(self, p: Sequence[float]) -> Tuple[float, float]:
        return (
            (p[0] + self.rail_in) * self.ppi,
            (p[1] + self.rail_in) * self.ppi,
        )

    def table_to_image(self, p: Sequence[float]) -> Tuple[float, float]:
        return self._project(self._ground_H, p)

    def ball_to_image(self, p: Sequence[float]) -> Tuple[float, float]:
        """Where the centre of a ball resting at table point ``p`` appears.

        A ball's centre is a radius above the cloth, so it appears nearer the
        horizon than the spot it rests on.  An earlier version of this simulator
        drew every ball at the image of that spot -- as if balls were discs
        painted on the cloth -- through a view no pinhole camera can produce.
        That hid a 2-4 inch position error the tracker made on every real clip.
        """
        return self._project(self._ball_H, p)

    @staticmethod
    def _project(H: np.ndarray, p: Sequence[float]) -> Tuple[float, float]:
        q = H @ np.array([float(p[0]), float(p[1]), 1.0])
        return (float(q[0] / q[2]), float(q[1] / q[2]))

    # -- static layers -----------------------------------------------------

    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        img[:] = (38, 52, 74)  # wooden rail
        x0 = int(self.rail_in * self.ppi)
        y0 = int(self.rail_in * self.ppi)
        x1 = int((self.rail_in + TABLE_L) * self.ppi)
        y1 = int((self.rail_in + TABLE_W) * self.ppi)
        cv2.rectangle(img, (x0, y0), (x1, y1), self.cloth, -1)

        # Cloth texture: low-amplitude noise so the mask is not trivially clean.
        noise = self.rng.normal(0, 3.2, (self.canvas_h, self.canvas_w, 1))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Pockets.
        for px, py in [
            (0, 0), (TABLE_L / 2, 0), (TABLE_L, 0),
            (0, TABLE_W), (TABLE_L / 2, TABLE_W), (TABLE_L, TABLE_W),
        ]:
            c = self.table_to_canvas((px, py))
            cv2.circle(img, (int(c[0]), int(c[1])), int(2.4 * self.ppi), (16, 16, 18), -1,
                       cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), (120, 150, 90), 2)
        return img

    #: Cushion nose height and top width, inches.
    CUSHION_H = 1.4
    CUSHION_W = 2.0

    def _draw_cushions(self, frame: np.ndarray) -> None:
        """The cushions, as raised blocks: a clothed top, and a face under the nose.

        Drawn the way the sample broadcasts show them.  The top of a cushion is
        covered in the same cloth as the bed and, lit from above, looks just
        like it -- which is why the calibrated outline of the cloth runs out
        over the cushion tops on a real table, and the nose is up to a couple
        of inches inside it.  What marks the nose is its face: undercut and in
        shadow, a dark band seen from in front (the far rail) and a thin dark
        line seen from the side (the long rails).  An earlier version drew
        the tops at 62% of the cloth's brightness, dark enough to fall outside
        the cloth, so the synthetic outline stopped at the nose and nothing
        ever tested the real case.
        """
        lift = self._plane_homography(self.CUSHION_H)

        def img(H: np.ndarray, pts) -> np.ndarray:
            q = np.array([[x, y, 1.0] for x, y in pts]) @ H.T
            return np.round(q[:, :2] / q[:, 2:3]).astype(np.int32)

        L, Wd, c = TABLE_L, TABLE_W, self.CUSHION_W
        face = tuple(int(v * 0.70) for v in self.cloth)
        top = tuple(int(v * 0.97) for v in self.cloth)
        shadow = tuple(int(v * 0.45) for v in self.cloth)
        # (nose from, nose to, outward direction) for each run of cushion.  A
        # pocket is a gap in the cushion, so the long rails are two runs each.
        g = 2.4  # pocket mouth half-width, as in the physics
        rails = [((0, g), (0, Wd - g), (-1, 0)), ((L, g), (L, Wd - g), (1, 0))]
        for y, out in ((0.0, -1), (Wd, 1)):
            rails.append(((g, y), (L / 2 - g, y), (0, out)))
            rails.append(((L / 2 + g, y), (L - g, y), (0, out)))
        centre = -self.R.T @ self.t
        # A shadow line has a physical width, so it is wider in a bigger
        # picture: drawn 2 px at every size, it thinned to a sub-pixel trace
        # once a 1080p clip was scaled to the tracker's 1280 px, and the
        # cushion tops ran into the bed.
        px = self.out_size[1] / 720.0
        wide, thin = max(1, int(round(2 * px))), max(1, int(round(px)))
        for a, b, out in rails:
            outer_a = (a[0] + out[0] * c, a[1] + out[1] * c)
            outer_b = (b[0] + out[0] * c, b[1] + out[1] * c)
            cv2.fillPoly(frame, [img(lift, [a, b, outer_b, outer_a])], top, cv2.LINE_AA)
            nose = img(lift, [a, b])
            # The face is visible only from the bed side of the nose.
            axis = 0 if out[0] else 1
            facing = (centre[axis] - a[axis]) * -(out[axis]) > 0
            if facing:
                foot = img(self._ground_H, [a, b])
                cv2.fillPoly(frame, [np.vstack([foot, nose[::-1]])], face, cv2.LINE_AA)
                cv2.line(frame, tuple(foot[0]), tuple(foot[1]), shadow, wide, cv2.LINE_AA)
            else:
                # Seen from behind, the nose is the edge of the top.
                cv2.line(frame, tuple(nose[0]), tuple(nose[1]), shadow, thin, cv2.LINE_AA)

    def _make_venue(self) -> np.ndarray:
        """Everything around the table: grey carpet, and a band of banners.

        The table no longer fills the frame once it is filmed like a
        broadcast, so what surrounds it matters.  A flat backdrop would be the
        most common colour in the frame and be measured as the cloth.  Real
        venues are grey carpet -- too grey to be cloth -- under walls of
        saturated sponsor banners, some the same family of colour as the cloth,
        which is exactly the hazard the cloth estimate has to survive.
        """
        w, h = self.out_size
        rng = np.random.default_rng(12345)
        grain = rng.normal(0.0, 7.0, (h // 4 + 1, w // 4 + 1, 1))
        grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
        venue = np.clip(np.full((h, w, 3), 62.0) + grain, 0, 255)
        banner_h = int(h * 0.16)
        colours = [(150, 60, 30), (40, 40, 40), (160, 80, 20), (30, 30, 150), (150, 60, 30)]
        width = w // len(colours) + 1
        for i, colour in enumerate(colours):
            venue[: banner_h, i * width:(i + 1) * width] = colour
        return venue.astype(np.uint8)

    def _make_vignette(self) -> np.ndarray:
        h, w = self.out_size[1], self.out_size[0]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.5, h * 0.38
        r = np.sqrt(((xx - cx) / (w * 0.75)) ** 2 + ((yy - cy) / (h * 0.9)) ** 2)
        # Bright under the lamp, falling off towards the corners: this is the
        # lighting gradient that breaks a single global HSV threshold.
        return np.clip(1.22 - 0.42 * r**1.6, 0.55, 1.3).astype(np.float32)[..., None]

    # -- per frame ---------------------------------------------------------

    def sphere_radius_px(self, p: Sequence[float], eps: float = 0.5) -> float:
        """Apparent radius of a ball at table point ``p``, in image pixels.

        A ball is a sphere, so its silhouette stays circular however obliquely
        the table is viewed -- its size follows the magnification *across* the
        line of sight, which is the largest singular value of the local
        table-to-image Jacobian.  Drawing balls as flat discs painted on the
        cloth (which an earlier version of this simulator did) squashes them
        under perspective and produces footage no real camera would record.
        """
        base = np.array(self.ball_to_image(p))
        dx = np.array(self.ball_to_image((p[0] + eps, p[1]))) - base
        dy = np.array(self.ball_to_image((p[0], p[1] + eps))) - base
        J = np.column_stack([dx / eps, dy / eps])
        singular = np.linalg.svd(J, compute_uv=False)
        return float(BALL_R * singular[0])

    def visibility(self, balls: Sequence[Ball]) -> Dict[str, float]:
        """How much of each ball the camera can see, 0..1.

        Balls are drawn back to front, so a ball is hidden wherever a nearer
        one's disc covers it.  From behind an end rail a static rack is seen
        almost along its axis and most of it is hidden this way: no tracker can
        report a ball that is not in the picture, so the evaluator does not
        ask it to (see ``evaluate.py --min-visible``).
        """
        active = [b for b in balls if b.active]
        discs = [(b.name, np.array(self.ball_to_image(b.pos)), self.sphere_radius_px(b.pos))
                 for b in active]
        spiral = np.array([
            (np.sqrt((i + 0.5) / 48.0) * np.cos(i * 2.39996), np.sqrt((i + 0.5) / 48.0) * np.sin(i * 2.39996))
            for i in range(48)
        ])
        out: Dict[str, float] = {}
        for name, centre, r in discs:
            pts = centre + spiral * r
            covered = np.zeros(len(pts), dtype=bool)
            for other, c2, r2 in discs:
                if other == name or c2[1] <= centre[1]:
                    continue  # only nearer balls (drawn later) hide this one
                covered |= np.linalg.norm(pts - c2, axis=1) < r2
            out[name] = 1.0 - float(covered.mean())
        return out

    def _draw_number_circle(
        self, frame: np.ndarray, b: Ball, centre: Tuple[float, float], r_px: int
    ) -> None:
        """The white circle a ball's number is printed in, turning as it rolls.

        A real solid is not one colour all over: its number circle is white,
        about two-fifths of the ball across, and comes and goes as the ball
        rolls.  That is exactly what a stripe test must not mistake for a
        stripe's caps, so the simulator has to show it.  The circle sits on
        the ball's surface, turning about the axis across its direction of
        travel by the distance rolled over the radius; it is drawn while it
        faces the camera, foreshortened as it turns away.
        """
        phi = b.rolled_in / BALL_R
        facing = math.cos(phi)
        if facing < 0.05:
            return
        ahead = self.ball_to_image(b.pos + b.heading)
        u = np.array(ahead) - np.array(centre)
        norm = float(np.linalg.norm(u))
        u = u / norm if norm > 1e-9 else np.array([1.0, 0.0])
        at = np.array(centre) + u * r_px * 0.7 * math.sin(phi)
        size = 0.4 * r_px
        axes = (max(1, int(round(size * facing))), max(1, int(round(size))))
        angle = math.degrees(math.atan2(u[1], u[0]))

        # Drawn into the ball's own disc only: near the limb the circle is
        # partly round the back of the ball.
        h, w = frame.shape[:2]
        x0, y0 = int(centre[0]) - r_px - 2, int(centre[1]) - r_px - 2
        x1, y1 = x0 + 2 * r_px + 5, y0 + 2 * r_px + 5
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return
        disc = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(disc, (int(round(centre[0])) - x0, int(round(centre[1])) - y0),
                   r_px - 1, 255, -1)
        mark = np.zeros_like(disc)
        cv2.ellipse(mark, (int(round(at[0])) - x0, int(round(at[1])) - y0), axes, angle,
                    0, 360, 255, -1)
        roi = frame[y0:y1, x0:x1]
        roi[(disc > 0) & (mark > 0)] = IVORY

    def render(self, balls: Sequence[Ball], cue_line: Optional[Tuple] = None) -> np.ndarray:
        canvas = self._background.copy()

        # Shadows are flat on the cloth, so they belong on the canvas and get
        # warped with it.
        shadow_r = int(round(BALL_R * self.ppi))
        # A shadow is a multiplicative darkening of whatever it falls on, not a
        # fixed dark colour.  Painting it as near-black (as an earlier version
        # did) produces a shadow far darker than any real one, which is not a
        # fair test of how a detector separates shadow from dark ball.
        shadow_colour = tuple(int(c * 0.72) for c in self.cloth)
        for b in balls:
            if not b.active:
                continue
            c = self.table_to_canvas(b.pos)
            cv2.circle(canvas, (int(round(c[0])) + 2, int(round(c[1])) + 3),
                       shadow_r, shadow_colour, -1, cv2.LINE_AA)

        if cue_line is not None:
            p0 = self.table_to_canvas(cue_line[0])
            p1 = self.table_to_canvas(cue_line[1])
            cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     (110, 160, 205), max(2, int(0.55 * self.ppi)), cv2.LINE_AA)

        frame = cv2.warpPerspective(
            canvas, self.H, self.out_size, dst=self._venue.copy(),
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        self._draw_cushions(frame)

        # Balls are drawn in image space, back to front, so a nearer ball
        # correctly occludes a farther one.
        visible = [b for b in balls if b.active]
        visible.sort(key=lambda b: self.ball_to_image(b.pos)[1])
        for b in visible:
            centre_f = self.ball_to_image(b.pos)
            centre = (int(round(centre_f[0])), int(round(centre_f[1])))
            r_px = max(2, int(round(self.sphere_radius_px(b.pos))))
            if b.striped:
                # A real striped ball is a ball of the cap colour carrying one
                # coloured band -- white caps on a standard set, black on the
                # TV set in the sample clips -- not a coloured ball with a
                # white band.  Rendering it the wrong way round makes every
                # stripe read as almost pure colour.
                cv2.circle(frame, centre, r_px, b.caps, -1, cv2.LINE_AA)
                cv2.ellipse(frame, centre, (r_px, max(1, int(r_px * 0.55))), 0,
                            0, 360, b.colour, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, centre, r_px, b.colour, -1, cv2.LINE_AA)
            if b.number:
                self._draw_number_circle(frame, b, centre_f, r_px)
            cv2.circle(frame, (centre[0] - r_px // 3, centre[1] - r_px // 3),
                       max(1, r_px // 4), (255, 255, 255), -1, cv2.LINE_AA)

        frame = (frame.astype(np.float32) * self._vignette)
        frame += self.rng.normal(0, 3.0, frame.shape)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(frame, (3, 3), 0.7)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------


def make_break_rack(seed: int = 0, ball_set: str = "standard") -> List[Ball]:
    """Cue ball behind the head string, the rest racked on the foot spot.

    A ball's name is its colour, with ``-stripe`` for a stripe: ``yellow`` is
    the 1 and ``yellow-stripe`` the 9.  (Until 23 Sep 2026 the stripes were
    picked by rack position, which made the 8 ball a white ball with a black
    band.  There is no such ball.)
    """
    rng = np.random.default_rng(seed)
    balls = [
        Ball(
            "cue",
            np.array([TABLE_L * 0.24, TABLE_W * 0.5 + rng.normal(0, 0.4)]),
            np.array([0.0, 0.0]),
            BALL_COLOURS["cue"],
        )
    ]

    apex = np.array([TABLE_L * 0.75, TABLE_W * 0.5])
    rack = RACKS[ball_set]
    spacing = 2 * BALL_R * 1.002
    idx = 0
    for row in range(4):
        for k in range(row + 1):
            if idx >= len(rack):
                break
            colour, number, striped = rack[idx]
            x = apex[0] + row * spacing * math.sqrt(3) / 2
            y = apex[1] + (k - row / 2.0) * spacing
            balls.append(
                Ball(
                    colour + ("-stripe" if striped else ""),
                    np.array([x, y]) + rng.normal(0, 0.02, 2),
                    np.array([0.0, 0.0]),
                    BALL_COLOURS[colour],
                    striped=striped,
                    number=number,
                    caps=CAPS[ball_set],
                )
            )
            idx += 1
    # Which way each ball's number faces to begin with, and the axis it turns
    # about until it first moves.
    for b in balls:
        b.rolled_in = float(rng.uniform(0.0, 2.0 * math.pi * BALL_R))
        heading = rng.uniform(0.0, 2.0 * math.pi)
        b.heading = np.array([math.cos(heading), math.sin(heading)])
    return balls


def generate(
    out_path: Path,
    *,
    seed: int = 0,
    fps: float = 30.0,
    duration_s: float = 6.0,
    approach_s: float = 0.8,
    width: int = 1280,
    height: int = 720,
    break_speed: float = 300.0,
    ground_truth_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    container_fps: Optional[float] = None,
    drop_rate: float = 0.0,
    capture_jitter: float = 0.0,
    camera: str = "end",
    ball_set: str = "standard",
    cloth: str = "broadcast",
) -> Dict[str, object]:
    """Simulate a break and write it as a video, with ground truth.

    ``fps`` is the rate the scene is filmed at.  With ``container_fps`` set, the
    clip is instead written the way the sample broadcast clips were evidently
    captured: a recorder running at ``container_fps`` grabs whatever frame is on
    screen, so a source frame is repeated in one, two or three slots, reaches
    the recorder a little late (``capture_jitter``, as a fraction of a source
    frame interval), and is occasionally never shown at all (``drop_rate``).
    The ground truth is then per *slot* -- where the balls are in the frame that
    slot actually shows -- with the true scene time alongside, so a tracker
    that believes the file's clock can be scored against the scene's.
    """
    balls = make_break_rack(seed, ball_set)
    sim = Simulation(balls, seed=seed)
    renderer = Renderer(out_size=(width, height), seed=seed, camera=camera,
                        cloth_bgr=CLOTHS[cloth])
    retimed = container_fps is not None
    file_fps = float(container_fps) if retimed else fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(out_path), fourcc, file_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    gt_rows: List[List[object]] = []
    dt = 1.0 / fps
    n_frames = int(duration_s * fps)
    n_approach = int(approach_s * fps)

    rng = np.random.default_rng(seed + 991)
    aim = np.array([1.0, rng.normal(0, 0.035)])
    aim /= np.linalg.norm(aim)

    def advance(f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Move the scene to source frame ``f``; return the cue stick, if drawn."""
        cue_line = None
        if f < n_approach:
            # Cue stick drawing back and striking: an elongated object that
            # the detector must not mistake for a ball.
            phase = f / max(1, n_approach)
            back = 9.0 * (1.0 - phase) + 2.0
            tip = balls[0].pos - aim * (BALL_R + back)
            butt = tip - aim * 46.0
            cue_line = (butt, tip)
        elif f == n_approach:
            balls[0].vel = aim * break_speed

        if f >= n_approach:
            sim.step(dt)
        return cue_line

    def rows_for(frame_index: int, t_file: float) -> List[List[object]]:
        rows: List[List[object]] = []
        seen = renderer.visibility(balls)
        for b in balls:
            if not b.active:
                continue
            img = renderer.ball_to_image(b.pos)
            rows.append(
                [
                    frame_index, round(t_file, 4), b.name,
                    round(float(b.pos[0]), 4), round(float(b.pos[1]), 4),
                    round(img[0], 2), round(img[1], 2),
                    round(float(np.linalg.norm(b.vel)), 3),
                    round(seen[b.name], 3),
                    b.number,
                ]
            )
        return rows

    #: (scene time, first frame of the file that shows it) for every source
    #: frame that reached the file, so an event can be pinned to the first
    #: frame in which its outcome is visible.
    shown_from: List[Tuple[float, int]] = []
    n_slots = 0

    try:
        if not retimed:
            for f in range(n_frames):
                cue_line = advance(f)
                frame = renderer.render(balls, cue_line)
                writer.write(frame)
                gt_rows.extend(rows_for(f, f / fps))
                shown_from.append((f / fps, f))
        else:
            cap = np.random.default_rng(seed + 7331)
            arrival = (
                np.arange(n_frames) + cap.uniform(0.0, capture_jitter, n_frames)
            ) / fps
            dropped = cap.random(n_frames) < drop_rate
            dropped[0] = False
            f = -1
            shown: Optional[np.ndarray] = None
            shown_rows: List[List[object]] = []
            shown_t = 0.0
            for s in range(int(duration_s * file_fps)):
                t_slot = s / file_fps
                while f + 1 < n_frames and arrival[f + 1] <= t_slot:
                    f += 1
                    cue_line = advance(f)
                    if dropped[f]:
                        continue
                    shown = renderer.render(balls, cue_line)
                    shown_t = f / fps
                    shown_rows = rows_for(-1, 0.0)
                    shown_from.append((shown_t, n_slots))
                if shown is None:
                    # Nothing has reached the recorder yet, so nothing is
                    # written -- and the ground truth must be numbered by the
                    # frames actually in the file, not by recorder slots, or
                    # it runs one frame behind the video from here on.
                    continue
                writer.write(shown)
                for row in shown_rows:
                    gt_rows.append(
                        [n_slots, round(n_slots / file_fps, 4)] + row[2:] + [round(shown_t, 4)]
                    )
                n_slots += 1
    finally:
        writer.release()

    if ground_truth_path is not None:
        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        with ground_truth_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            header = ["frame", "t_s", "ball", "x_in", "y_in", "x_px", "y_px", "speed_in_s",
                      "visible", "number"]
            if retimed:
                header.append("scene_t_s")
            w.writerow(header)
            w.writerows(gt_rows)

    # The simulation clock starts when the cue ball is struck, which is one
    # source frame before the first frame that shows it moving.
    t0 = (n_approach - 1) / fps
    events = []
    for e in sim.events:
        t_scene = float(e["t_s"]) + t0
        first = next((slot for t, slot in shown_from if t >= t_scene - 1e-9), None)
        events.append({**e, "t_s": round(t_scene, 5), "frame": first})
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps(events, indent=1) + "\n", encoding="utf-8")

    return {
        "video": str(out_path),
        "ground_truth": str(ground_truth_path) if ground_truth_path else None,
        "events": str(events_path) if events_path else None,
        "frames": n_slots if retimed else n_frames,
        "fps": file_fps,
        "scene_fps": fps,
        "balls": len(balls),
        "potted": sum(1 for b in balls if not b.active),
        "event_counts": {
            kind: sum(1 for e in events if e["type"] == kind)
            for kind in ("collision", "cushion", "pot")
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="assets/synthetic_break.mp4")
    p.add_argument("--ground-truth", default=None,
                   help="CSV of exact ball positions per frame")
    p.add_argument("--events", default=None,
                   help="JSON of every collision, cushion contact and pot the "
                        "physics resolved")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0,
                   help="rate the scene is filmed at")
    p.add_argument("--container-fps", type=float, default=None,
                   help="write the clip at this rate instead, repeating and "
                        "occasionally dropping source frames the way a screen "
                        "recording of a broadcast does (e.g. --fps 25 "
                        "--container-fps 37.5)")
    p.add_argument("--drop-rate", type=float, default=0.0,
                   help="with --container-fps: fraction of source frames never shown")
    p.add_argument("--capture-jitter", type=float, default=0.0,
                   help="with --container-fps: how late a source frame may reach "
                        "the recorder, as a fraction of its interval")
    p.add_argument("--camera", choices=sorted(CAMERAS), default="end",
                   help="end: behind an end rail, like the sample broadcasts; "
                        "side: across the table from a long rail; overhead: "
                        "straight down from the ceiling; corner: a tripod at a "
                        "corner of the room")
    p.add_argument("--ball-set", choices=sorted(RACKS), default="standard",
                   help="standard: 4 purple, 5 orange, white-capped stripes; tv: "
                        "the sample broadcasts' set (4 pink, 5 purple, black-capped "
                        "stripes), racked 1-9")
    p.add_argument("--cloth", choices=sorted(CLOTHS), default="broadcast",
                   help="broadcast: the sample clips' blue-grey cloth; green: "
                        "the simulator's cloth until 23 Sep 2026; blue, red, tan, "
                        "grey: other cloths")
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--break-speed", type=float, default=300.0)
    args = p.parse_args()

    gt = Path(args.ground_truth) if args.ground_truth else None
    info = generate(
        Path(args.out), seed=args.seed, fps=args.fps, duration_s=args.duration,
        width=args.width, height=args.height, break_speed=args.break_speed,
        ground_truth_path=gt,
        events_path=Path(args.events) if args.events else None,
        container_fps=args.container_fps, drop_rate=args.drop_rate,
        capture_jitter=args.capture_jitter, camera=args.camera,
        ball_set=args.ball_set, cloth=args.cloth,
    )
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
