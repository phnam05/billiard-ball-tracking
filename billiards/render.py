"""Drawing: annotated camera view plus a synthetic overhead diagram.

The old overhead view re-warped the camera pixels every frame, which is blurry
at the far cushion (those pixels are heavily stretched), slow, and re-derived the
table corners from scratch on every single frame.  Because tracking now happens
in table coordinates anyway, the overhead panel here is *drawn* rather than
warped: crisp at any zoom, free to render, and it shows the tracker's actual
belief rather than a picture of it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .detect import Detection
from .events import Event, EventType
from .geometry import TableModel
from .table import ClothModel
from .track import Track, TrackState


FONT = cv2.FONT_HERSHEY_SIMPLEX

EVENT_COLOURS: Dict[EventType, Tuple[int, int, int]] = {
    EventType.COLLISION: (60, 255, 255),
    EventType.CUSHION: (255, 200, 80),
    EventType.POT: (90, 90, 255),
    EventType.BALL_STRUCK: (150, 255, 150),
}


def _contrast_colour(bgr: Sequence[int]) -> Tuple[int, int, int]:
    luma = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (20, 20, 20) if luma > 140 else (245, 245, 245)


def overhead_flips(table: TableModel) -> Tuple[bool, bool]:
    """Which table axes the top-down diagram mirrors, so it matches the camera.

    Table coordinates are laid out on the diagram with the long axis across,
    but which corner is the origin is an accident of calibration.  Drawn as-is,
    the diagram of a broadcast shot from behind an end rail came out as a
    *reflection* of the picture above it: the far rail on the left and the
    right-hand balls at the bottom, so every ball sat on the wrong side of the
    table from where the viewer could see it.  A rotation is easy to read
    across; a mirror image is not.

    So of the four ways to lay the table down with its long axis across, keep
    the ones that are a rotation of the camera view, and of those the one
    turned least.  When two are turned equally -- a camera looking down the
    table, 90 degrees either way -- the far end goes on the right, so the
    diagram reads away from the viewer the way the picture does upwards.
    """
    L, W = table.length_in, table.width_in
    centre = np.array([L / 2.0, W / 2.0])
    step = 0.1 * min(L, W)
    img = table.table_to_image(
        np.array([centre, centre + (step, 0.0), centre + (0.0, step)])
    )
    # Image directions of table +x and +y, as the columns of E.
    E = np.column_stack([img[1] - img[0], img[2] - img[0]])
    if not np.all(np.isfinite(E)) or abs(np.linalg.det(E)) < 1e-9:
        return False, False
    E_inv = np.linalg.inv(E)

    best = None
    for flip_x in (False, True):
        for flip_y in (False, True):
            # image -> diagram, as a linear map at the centre of the table.
            A = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0]) @ E_inv
            if np.linalg.det(A) <= 0:
                continue  # a reflection
            angle = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            # Least turned first; at a tie, "up in the picture" goes right
            # (+90 degrees, clockwise on screen) rather than left.
            key = (round(abs(angle) / 5.0), -angle)
            if best is None or key < best[0]:
                best = (key, (flip_x, flip_y))
    return best[1] if best else (False, False)


class Renderer:
    def __init__(self, cfg: Config, table: TableModel, cloth: Optional[ClothModel] = None) -> None:
        self.cfg = cfg
        self.table = table
        self.cloth = cloth
        self._cloth_bgr = self._cloth_display_colour()
        self._flip_x, self._flip_y = overhead_flips(table)

    def _cloth_display_colour(self) -> Tuple[int, int, int]:
        if self.cloth is None:
            return (70, 110, 60)
        hsv = np.uint8([[[int(self.cloth.hue) % 180,
                         int(np.clip(self.cloth.sat, 0, 255)),
                         int(np.clip(self.cloth.val * 0.75, 0, 255))]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # ------------------------------------------------------------------
    # Camera view
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        recent_events: Sequence[Event],
        hud: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        out = frame.copy()
        r = self.cfg.render

        if r.draw_table_outline:
            self._draw_table_outline(out)
        if r.draw_detections:
            self._draw_detections(out, detections)
        if r.draw_trajectories:
            self._draw_trails(out, tracks)
        self._draw_balls(out, tracks)
        if r.draw_events:
            self._draw_events(out, recent_events)
        return self._compose(out, tracks, recent_events, hud)

    def compose_idle(
        self, frame: np.ndarray, hud: Optional[Dict[str, object]] = None
    ) -> np.ndarray:
        """The canvas for a frame with nothing to report.

        Used while the calibrated table is off screen.  It goes through the
        same composition as a tracked frame so that every frame of the output
        video is the same size -- the writer adopts the first frame's size and
        would otherwise have to squash the rest.
        """
        return self._compose(frame.copy(), [], [], hud)

    # ------------------------------------------------------------------
    # Composition: the picture, plus whatever is drawn beside it
    # ------------------------------------------------------------------

    def _compose(
        self,
        img: np.ndarray,
        tracks: Sequence[Track],
        events: Sequence[Event],
        hud: Optional[Dict[str, object]],
    ) -> np.ndarray:
        r = self.cfg.render
        show_hud = bool(r.draw_hud and hud)
        if r.overhead_panel_place == "below":
            if not r.overhead_panel and not show_hud:
                return img
            return self._with_panel_bar(img, tracks, events, hud if show_hud else None)

        if r.overhead_panel:
            self._blit_overhead(img, tracks, events)
        if r.draw_hud and hud:
            self.draw_hud(img, hud)
        return img

    def _with_panel_bar(
        self,
        img: np.ndarray,
        tracks: Sequence[Track],
        events: Sequence[Event],
        hud: Optional[Dict[str, object]],
    ) -> np.ndarray:
        """Stack the picture on a bar holding the diagram and the status text.

        Everything synthetic ends up in the bar, so the frame above it is the
        broadcast picture with only the tracker's own marks on the balls --
        which is what someone checking the tracking actually wants to look at.
        """
        h, w = img.shape[:2]
        pad = 10

        panel = None
        if self.cfg.render.overhead_panel:
            panel = self.overhead(tracks, events)
            target_w = max(40, int(w * self.cfg.render.overhead_panel_scale))
            scale = target_w / panel.shape[1]
            panel = cv2.resize(
                panel, (target_w, max(1, int(round(panel.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        text_h = self._status_height(hud) if hud else 0
        content_h = max(panel.shape[0] if panel is not None else 0, text_h)
        bar_h = content_h + 2 * pad

        out = np.empty((h + bar_h, w, 3), dtype=img.dtype)
        out[:h] = img
        out[h:] = (22, 22, 22)
        cv2.line(out, (0, h), (w, h), (70, 70, 70), 1)

        x0, y0 = pad, h + pad
        if panel is not None:
            ph, pw = panel.shape[:2]
            out[y0 : y0 + ph, x0 : x0 + pw] = panel
            cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph),
                          (110, 110, 110), 1)
            x0 += pw + 2 * pad

        if hud:
            self._draw_status(out, hud, x0, y0)
        return out

    def _status_line_height(self) -> int:
        scale = self.cfg.render.font_scale * 1.15
        return cv2.getTextSize("Ag", FONT, scale, 1)[0][1] + 10

    def _status_height(self, hud: Dict[str, object]) -> int:
        return self._status_line_height() * len(hud)

    def _draw_status(
        self, img: np.ndarray, hud: Dict[str, object], x: int, y: int
    ) -> None:
        """The status lines, laid out in the bar beside the diagram."""
        scale = self.cfg.render.font_scale * 1.15
        lines = [(str(k), str(v)) for k, v in hud.items()]
        if not lines:
            return
        key_w = max(cv2.getTextSize(k, FONT, scale, 1)[0][0] for k, _ in lines)
        line_h = self._status_line_height()

        cursor = y + line_h - 4
        for key, value in lines:
            if cursor > img.shape[0] - 4:
                break
            cv2.putText(img, key, (x, cursor), FONT, scale,
                        (130, 130, 130), 1, cv2.LINE_AA)
            cv2.putText(img, value, (x + key_w + 12, cursor), FONT, scale,
                        (235, 235, 235), 1, cv2.LINE_AA)
            cursor += line_h

    def _draw_table_outline(self, img: np.ndarray) -> None:
        poly = self.table.bed_polygon_image(0.0).astype(np.int32)
        cv2.polylines(img, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
        margin = self.cfg.table.bed_margin_ball_diameters * self.table.ball_diameter_in
        inner = self.table.bed_polygon_image(margin).astype(np.int32)
        cv2.polylines(img, [inner], True, (140, 140, 140), 1, cv2.LINE_AA)

    def _draw_detections(self, img: np.ndarray, detections: Sequence[Detection]) -> None:
        for d in detections:
            c = (int(round(d.centre_image[0])), int(round(d.centre_image[1])))
            cv2.circle(img, c, int(round(d.radius_px)), (0, 200, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(img, c, (0, 200, 255), cv2.MARKER_CROSS, 6, 1)

    def _draw_trails(self, img: np.ndarray, tracks: Sequence[Track]) -> None:
        thickness = max(1, self.cfg.render.trail_thickness)
        for track in tracks:
            if len(track.trail) < 2:
                continue
            base = track.signature.bgr
            pts = [s.image_xy for s in track.trail]
            n = len(pts)
            for i in range(1, n):
                # Fade towards the cloth colour with age, so the newest part of
                # the path reads as the strongest line.
                w = i / float(n)
                colour = tuple(
                    int(base[k] * (0.25 + 0.75 * w) + 255 * 0.15 * w) for k in range(3)
                )
                p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                cv2.line(img, p0, p1, colour, thickness, cv2.LINE_AA)

    def _draw_balls(self, img: np.ndarray, tracks: Sequence[Track]) -> None:
        r = self.cfg.render
        for track in tracks:
            pos = track.kf.position
            img_pt = self.table.ball_table_to_image([tuple(pos)])[0]
            centre = (int(round(img_pt[0])), int(round(img_pt[1])))
            radius = max(3, int(round(self.table.expected_ball_radius_px(tuple(img_pt)))))
            colour = track.signature.bgr

            coasting = track.state is TrackState.COASTING
            # A coasted ball is the tracker's guess, not an observation: dashed
            # rather than solid, so a viewer can tell belief from evidence.
            if coasting:
                self._dashed_circle(img, centre, radius, colour)
            else:
                cv2.circle(img, centre, radius, colour, 2, cv2.LINE_AA)
                cv2.circle(img, centre, radius, (25, 25, 25), 1, cv2.LINE_AA)

            if r.draw_ids:
                label = track.label
                scale = r.font_scale
                (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
                box_tl = (centre[0] - tw // 2 - 3, centre[1] - radius - th - 7)
                box_br = (centre[0] + tw // 2 + 3, centre[1] - radius - 2)
                cv2.rectangle(img, box_tl, box_br, colour, -1)
                cv2.putText(
                    img, label,
                    (centre[0] - tw // 2, centre[1] - radius - 5),
                    FONT, scale, _contrast_colour(colour), 1, cv2.LINE_AA,
                )

    @staticmethod
    def _dashed_circle(
        img: np.ndarray, centre: Tuple[int, int], radius: int,
        colour: Tuple[int, int, int], segments: int = 12,
    ) -> None:
        for k in range(segments):
            if k % 2:
                continue
            a0 = 360.0 * k / segments
            a1 = 360.0 * (k + 1) / segments
            cv2.ellipse(img, centre, (radius, radius), 0, a0, a1, colour, 2, cv2.LINE_AA)

    def _draw_events(self, img: np.ndarray, events: Sequence[Event]) -> None:
        for e in events:
            colour = EVENT_COLOURS.get(e.type, (255, 255, 255))
            c = (int(round(e.image_xy[0])), int(round(e.image_xy[1])))
            if e.type is EventType.COLLISION:
                cv2.circle(img, c, 9, colour, 2, cv2.LINE_AA)
                cv2.drawMarker(img, c, colour, cv2.MARKER_TILTED_CROSS, 14, 2)
            elif e.type is EventType.CUSHION:
                cv2.drawMarker(img, c, colour, cv2.MARKER_DIAMOND, 14, 2)
            elif e.type is EventType.POT:
                cv2.circle(img, c, 12, colour, 2, cv2.LINE_AA)
                cv2.putText(img, "POT", (c[0] + 14, c[1] + 4), FONT,
                            self.cfg.render.font_scale, colour, 1, cv2.LINE_AA)
            else:
                cv2.drawMarker(img, c, colour, cv2.MARKER_TRIANGLE_UP, 12, 2)

    def draw_hud(self, img: np.ndarray, hud: Dict[str, object]) -> None:
        """Overlay the status box.  Public because the pipeline draws it on its
        own when tracking is paused and there is nothing else to render."""
        lines = [f"{k}: {v}" for k, v in hud.items()]
        scale = self.cfg.render.font_scale
        pad = 6
        heights = []
        widths = []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, FONT, scale, 1)
            widths.append(tw)
            heights.append(th)
        if not lines:
            return
        box_w = max(widths) + 2 * pad
        line_h = max(heights) + 6
        box_h = line_h * len(lines) + pad

        overlay = img.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.rectangle(img, (8, 8), (8 + box_w, 8 + box_h), (90, 90, 90), 1)

        y = 8 + pad + heights[0]
        for line in lines:
            cv2.putText(img, line, (8 + pad, y), FONT, scale,
                        (235, 235, 235), 1, cv2.LINE_AA)
            y += line_h

    # ------------------------------------------------------------------
    # Overhead diagram
    # ------------------------------------------------------------------

    def overhead(
        self,
        tracks: Sequence[Track],
        events: Sequence[Event] = (),
        px_per_inch: Optional[float] = None,
    ) -> np.ndarray:
        """Synthetic top-down view rendered straight from table coordinates."""
        ppi = px_per_inch or self.cfg.table.overhead_px_per_inch
        L, W = self.table.length_in, self.table.width_in
        pad = int(round(2.5 * self.table.ball_diameter_in * ppi))
        w = int(round(L * ppi)) + 2 * pad
        h = int(round(W * ppi)) + 2 * pad

        img = np.full((h, w, 3), 24, dtype=np.uint8)
        # Rail
        cv2.rectangle(img, (pad // 3, pad // 3), (w - pad // 3, h - pad // 3),
                      (35, 45, 60), -1)
        # Bed
        cv2.rectangle(img, (pad, pad), (w - pad, h - pad), self._cloth_bgr, -1)
        cv2.rectangle(img, (pad, pad), (w - pad, h - pad), (200, 200, 200), 1)

        flip_x, flip_y = self._flip_x, self._flip_y

        def to_px(p: Sequence[float]) -> Tuple[int, int]:
            x = L - p[0] if flip_x else p[0]
            y = W - p[1] if flip_y else p[1]
            return (int(round(pad + x * ppi)), int(round(pad + y * ppi)))

        # Pockets
        pocket_r = int(round(self.table.ball_diameter_in * ppi * 0.85))
        for pk in self.table.pockets_table():
            cv2.circle(img, to_px(pk), pocket_r, (12, 12, 12), -1, cv2.LINE_AA)

        # Head string and foot spot, useful landmarks for reading a break.
        cv2.line(img, to_px((L * 0.25, 0.0)), to_px((L * 0.25, W)), (190, 190, 190), 1,
                 cv2.LINE_AA)
        cv2.circle(img, to_px((L * 0.75, W / 2.0)), 2, (220, 220, 220), -1, cv2.LINE_AA)

        ball_r = max(2, int(round(self.table.ball_radius_in * ppi)))

        for track in tracks:
            if len(track.trail) >= 2:
                pts = np.array([to_px(s.table_xy) for s in track.trail], dtype=np.int32)
                base = track.signature.bgr
                n = len(pts)
                for i in range(1, n):
                    wgt = i / float(n)
                    colour = tuple(int(base[k] * (0.3 + 0.7 * wgt)) for k in range(3))
                    cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), colour,
                             max(1, self.cfg.render.trail_thickness), cv2.LINE_AA)

        for track in tracks:
            centre = to_px(track.kf.position)
            colour = track.signature.bgr
            if track.state is TrackState.COASTING:
                cv2.circle(img, centre, ball_r, colour, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, centre, ball_r, colour, -1, cv2.LINE_AA)
                cv2.circle(img, centre, ball_r, (20, 20, 20), 1, cv2.LINE_AA)

        for e in events:
            cv2.drawMarker(img, to_px(e.table_xy),
                           EVENT_COLOURS.get(e.type, (255, 255, 255)),
                           cv2.MARKER_TILTED_CROSS, max(6, ball_r * 2), 1)
        return img

    def _blit_overhead(
        self, img: np.ndarray, tracks: Sequence[Track], events: Sequence[Event]
    ) -> None:
        panel = self.overhead(tracks, events)
        target_w = int(img.shape[1] * self.cfg.render.overhead_panel_scale)
        if target_w < 40:
            return
        scale = target_w / panel.shape[1]
        panel = cv2.resize(panel, (target_w, max(1, int(panel.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        ph, pw = panel.shape[:2]
        if ph >= img.shape[0] or pw >= img.shape[1]:
            return
        margin = 10
        y0 = img.shape[0] - ph - margin
        x0 = margin
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + pw + 2, y0 + ph + 2),
                      (200, 200, 200), 1)
        img[y0 : y0 + ph, x0 : x0 + pw] = panel


def debug_panel(masks: Dict[str, np.ndarray], width: int = 960) -> np.ndarray:
    """Stack named single-channel masks into one labelled inspection image."""
    tiles = []
    for name, mask in masks.items():
        if mask is None:
            continue
        vis = mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        scale = width / vis.shape[1]
        vis = cv2.resize(vis, (width, max(1, int(vis.shape[0] * scale))))
        cv2.putText(vis, name, (10, 24), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(vis)
    if not tiles:
        return np.zeros((64, width, 3), dtype=np.uint8)
    return np.vstack(tiles)
