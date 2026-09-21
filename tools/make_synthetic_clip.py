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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TABLE_L = 100.0  # inches, playing surface
TABLE_W = 50.0
BALL_R = 1.125

# Standard-ish pool ball colours, in BGR.
BALL_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "cue": (248, 248, 250),
    "yellow": (40, 210, 240),
    "blue": (200, 70, 30),
    "red": (45, 45, 215),
    "purple": (140, 50, 110),
    "orange": (30, 130, 245),
    "green": (60, 140, 55),
    "maroon": (50, 45, 130),
    "black": (24, 24, 26),
}


@dataclass
class Ball:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    colour: Tuple[int, int, int]
    striped: bool = False
    active: bool = True


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

    def step(self, dt: float, substeps: int = 8) -> None:
        h = dt / substeps
        for _ in range(substeps):
            self._substep(h)

    def _substep(self, h: float) -> None:
        decay = math.exp(-h / self.tau)
        for b in self.balls:
            if not b.active:
                continue
            b.pos = b.pos + b.vel * self.tau * (1.0 - decay)
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
                    b.vel[axis] = -b.vel[axis] * self.restitution
                elif b.pos[axis] > limit - BALL_R:
                    b.pos[axis] = (limit - BALL_R) - (b.pos[axis] - (limit - BALL_R))
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

    @property
    def moving(self) -> bool:
        return any(
            b.active and float(np.linalg.norm(b.vel)) > 0.5 for b in self.balls
        )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class Renderer:
    """Render the table overhead, then warp it through a virtual camera."""

    def __init__(
        self,
        out_size: Tuple[int, int] = (1280, 720),
        px_per_inch: float = 11.0,
        cloth_bgr: Tuple[int, int, int] = (86, 122, 46),
        tilt: float = 0.34,
        yaw: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.out_size = out_size
        self.ppi = px_per_inch
        self.cloth = cloth_bgr
        self.rng = np.random.default_rng(seed)

        self.rail_in = 5.0
        self.canvas_w = int((TABLE_L + 2 * self.rail_in) * px_per_inch)
        self.canvas_h = int((TABLE_W + 2 * self.rail_in) * px_per_inch)
        self.H = self._camera_homography(tilt, yaw)
        self._background = self._make_background()
        self._vignette = self._make_vignette()

    # -- geometry ----------------------------------------------------------

    def _camera_homography(self, tilt: float, yaw: float) -> np.ndarray:
        """Map the flat canvas to a trapezoid, mimicking a camera looking down
        the table from behind one end rail."""
        w, h = self.out_size
        src = np.float32(
            [[0, 0], [self.canvas_w, 0], [self.canvas_w, self.canvas_h], [0, self.canvas_h]]
        )
        inset = w * tilt * 0.5
        shift = w * yaw
        top_y = h * 0.20
        bot_y = h * 0.94
        dst = np.float32(
            [
                [inset + shift, top_y],
                [w - inset + shift, top_y],
                [w * 0.99, bot_y],
                [w * 0.01, bot_y],
            ]
        )
        return cv2.getPerspectiveTransform(src, dst)

    def table_to_canvas(self, p: Sequence[float]) -> Tuple[float, float]:
        return (
            (p[0] + self.rail_in) * self.ppi,
            (p[1] + self.rail_in) * self.ppi,
        )

    def table_to_image(self, p: Sequence[float]) -> Tuple[float, float]:
        c = np.array([[self.table_to_canvas(p)]], dtype=np.float32)
        out = cv2.perspectiveTransform(c, self.H)[0, 0]
        return (float(out[0]), float(out[1]))

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
        base = np.array(self.table_to_image(p))
        dx = np.array(self.table_to_image((p[0] + eps, p[1]))) - base
        dy = np.array(self.table_to_image((p[0], p[1] + eps))) - base
        J = np.column_stack([dx / eps, dy / eps])
        singular = np.linalg.svd(J, compute_uv=False)
        return float(BALL_R * singular[0])

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
            canvas, self.H, self.out_size, borderValue=(28, 30, 34)
        )

        # Balls are drawn in image space, back to front, so a nearer ball
        # correctly occludes a farther one.
        visible = [b for b in balls if b.active]
        visible.sort(key=lambda b: self.table_to_image(b.pos)[1])
        for b in visible:
            centre_f = self.table_to_image(b.pos)
            centre = (int(round(centre_f[0])), int(round(centre_f[1])))
            r_px = max(2, int(round(self.sphere_radius_px(b.pos))))
            cv2.circle(frame, centre, r_px, b.colour, -1, cv2.LINE_AA)
            if b.striped:
                cv2.ellipse(frame, centre, (r_px, max(1, int(r_px * 0.42))), 0,
                            0, 360, (250, 250, 250), -1, cv2.LINE_AA)
                cv2.circle(frame, centre, r_px, b.colour, 1, cv2.LINE_AA)
            cv2.circle(frame, (centre[0] - r_px // 3, centre[1] - r_px // 3),
                       max(1, r_px // 4), (255, 255, 255), -1, cv2.LINE_AA)

        frame = (frame.astype(np.float32) * self._vignette)
        frame += self.rng.normal(0, 3.0, frame.shape)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(frame, (3, 3), 0.7)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------


def make_break_rack(seed: int = 0) -> List[Ball]:
    """Cue ball behind the head string, eight balls racked on the foot spot."""
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
    names = ["yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"]
    spacing = 2 * BALL_R * 1.002
    idx = 0
    for row in range(4):
        for k in range(row + 1):
            if idx >= len(names):
                break
            x = apex[0] + row * spacing * math.sqrt(3) / 2
            y = apex[1] + (k - row / 2.0) * spacing
            balls.append(
                Ball(
                    names[idx],
                    np.array([x, y]) + rng.normal(0, 0.02, 2),
                    np.array([0.0, 0.0]),
                    BALL_COLOURS[names[idx]],
                    striped=(idx % 3 == 1),
                )
            )
            idx += 1
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
) -> Dict[str, object]:
    balls = make_break_rack(seed)
    sim = Simulation(balls, seed=seed)
    renderer = Renderer(out_size=(width, height), seed=seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.suffix == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    gt_rows: List[List[object]] = []
    dt = 1.0 / fps
    n_frames = int(duration_s * fps)
    n_approach = int(approach_s * fps)

    rng = np.random.default_rng(seed + 991)
    aim = np.array([1.0, rng.normal(0, 0.035)])
    aim /= np.linalg.norm(aim)

    try:
        for f in range(n_frames):
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

            frame = renderer.render(balls, cue_line)
            writer.write(frame)

            for b in balls:
                if not b.active:
                    continue
                img = renderer.table_to_image(b.pos)
                gt_rows.append(
                    [
                        f, round(f / fps, 4), b.name,
                        round(float(b.pos[0]), 4), round(float(b.pos[1]), 4),
                        round(img[0], 2), round(img[1], 2),
                        round(float(np.linalg.norm(b.vel)), 3),
                    ]
                )
    finally:
        writer.release()

    if ground_truth_path is not None:
        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        with ground_truth_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t_s", "ball", "x_in", "y_in", "x_px", "y_px", "speed_in_s"])
            w.writerows(gt_rows)

    return {
        "video": str(out_path),
        "ground_truth": str(ground_truth_path) if ground_truth_path else None,
        "frames": n_frames,
        "fps": fps,
        "balls": len(balls),
        "potted": sum(1 for b in balls if not b.active),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="assets/synthetic_break.mp4")
    p.add_argument("--ground-truth", default=None,
                   help="CSV of exact ball positions per frame")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
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
    )
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
