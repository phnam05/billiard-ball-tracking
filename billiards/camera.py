"""How far the camera has moved since a view of the table was fitted.

A broadcast camera pushes in and pans during play, and a tripod gets nudged.
Refitting the table's outline to follow it works only while the outline can
be seen: on the 2026 Premier League final the camera pushed in while the
player was down on the shot, his body over the near rail, and every outline
fitted then cut a corner -- so the table stayed where it had been, a few
inches off, while the balls were tracked on it.

What stays visible is the scenery round the table: rails, sights, pockets,
the balls at rest, the venue behind.  Matching image features (ORB) between
a picture taken when the view was fitted and the current frame, and fitting
one homography to the matches with RANSAC, measures the camera's motion
directly.  The player moves and is left out as an outlier; so are the
balls that moved.  The table's corners are then carried along by it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import TableModel

#: Pictures are matched at this width; ORB at 640 px takes about 5 ms.
_WORK_WIDTH = 640
#: Features are taken from round the table: the bed and this many inches
#: beyond its cushions, over the rails, where the sights and pockets are.
_AROUND_IN = 14.0
#: Broadcast graphics (the score bar, a logo) stay still on screen while the
#: camera moves, and would vote for "no motion".  They live in bands along
#: the top and bottom of the picture, which are left out.
_TOP_BAND = 0.07
_BOTTOM_BAND = 0.16
#: Lowe's ratio test, and how many matches must agree with the motion.
_RATIO = 0.75
_MIN_INLIERS = 30
_MIN_INLIER_SHARE = 0.35


def _orb() -> "cv2.ORB":
    return cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=12)


def _prepare(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = min(1.0, _WORK_WIDTH / float(frame.shape[1]))
    small = frame if scale == 1.0 else cv2.resize(
        frame, (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return grey, scale


def _bands_mask(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    mask[: int(round(h * _TOP_BAND))] = 0
    mask[h - int(round(h * _BOTTOM_BAND)):] = 0
    return mask


class Keyframe:
    """A picture of the table from one view, to measure camera motion against."""

    def __init__(self, frame: np.ndarray, table: TableModel) -> None:
        grey, self.scale = _prepare(frame)
        mask = _bands_mask(grey.shape)
        around = table.table_to_image(table.bed_polygon_table(-_AROUND_IN)) * self.scale
        region = np.zeros_like(mask)
        if np.all(np.isfinite(around)):
            lim = 4.0 * max(grey.shape)
            cv2.fillConvexPoly(region, np.round(np.clip(around, -lim, lim)).astype(np.int32), 255)
        self.points, self.descriptors = _orb().detectAndCompute(grey, cv2.bitwise_and(mask, region))
        self.size = (frame.shape[1], frame.shape[0])

    @property
    def usable(self) -> bool:
        return self.descriptors is not None and len(self.points) >= _MIN_INLIERS

    def motion_to(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """The homography taking this keyframe's pixels to ``frame``'s, or None.

        None when too few features agree on one motion: another camera, a
        close-up, a replay graphic -- anything but this view moved a little.
        """
        if not self.usable or (frame.shape[1], frame.shape[0]) != self.size:
            return None
        grey, scale = _prepare(frame)
        points, descriptors = _orb().detectAndCompute(grey, _bands_mask(grey.shape))
        if descriptors is None or len(points) < _MIN_INLIERS:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < _RATIO * p[1].distance]
        if len(good) < _MIN_INLIERS:
            return None
        src = np.float32([self.points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([points[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
        if H is None or inliers is None:
            return None
        count = int(inliers.sum())
        if count < _MIN_INLIERS or count < _MIN_INLIER_SHARE * len(good):
            return None
        # Back to full-size pixels at both ends.
        down = np.diag([self.scale, self.scale, 1.0])
        up = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        return up @ H @ down


def carried(table: TableModel, motion: np.ndarray) -> Optional[TableModel]:
    """``table`` as the camera sees it after ``motion``, or None if that fails.

    The corners keep their order, so (0, 0) stays at the same corner of the
    table and a ball keeps its coordinates.  That is checked: the new model
    must put points of the table where the old one, moved, does -- it decides
    for itself which image axis is the table's length, and a view that moved
    far enough could decide differently.
    """
    def move(pts: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), motion).reshape(-1, 2)

    try:
        moved = TableModel(
            corners_image=move(table.corners_image),
            length_in=table.length_in,
            width_in=table.width_in,
            ball_diameter_in=table.ball_diameter_in,
            has_pockets=table.has_pockets,
            image_size=table.image_size,
            ball_parallax=table.ball_parallax,
            outline_image=None if table.outline_image is None else move(table.outline_image),
            mirrored=table.mirrored,
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return None
    L, W = table.length_in, table.width_in
    probe = np.array([[0.2 * L, 0.3 * W], [0.8 * L, 0.3 * W], [0.5 * L, 0.7 * W], [0.3 * L, 0.8 * W]])
    back = moved.image_to_table(move(table.table_to_image(probe)))
    if not np.all(np.isfinite(back)) or float(np.max(np.linalg.norm(back - probe, axis=1))) > 1.0:
        return None
    return moved
