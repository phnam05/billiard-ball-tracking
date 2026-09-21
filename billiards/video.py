"""Video input, output and data export.

Three bugs in the original are fixed here, all of which silently corrupted the
saved video:

* the writer was created with the *source* frame size but fed frames that had
  been resized to 854x480, so the file it produced was unplayable;
* the frame was written **before** any annotation was drawn, so even a valid
  file would have contained no tracking overlay;
* the frame rate was hard-coded to 10 regardless of the source, so playback ran
  at the wrong speed and every timestamp derived from it was wrong.

The writer below takes its size from the first frame it is actually given and
its rate from the source clip.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 2),
        }


def probe(path: Union[str, Path]) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1e-3:
            fps = 30.0  # some containers simply do not report it
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / float(frame.shape[1])
    return cv2.resize(
        frame,
        (max_width, max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def read_frames(
    path: Union[str, Path],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_width: int = 0,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for the requested range."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        idx = start_frame
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, resize_to_width(frame, max_width)
            idx += 1
    finally:
        cap.release()


def sample_frames(
    path: Union[str, Path],
    count: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_width: int = 0,
) -> List[np.ndarray]:
    """Grab ``count`` frames spread evenly over the clip, for calibration.

    Spreading the sample matters: calibrating on the first frame alone means a
    caption bar, a player bending over the rail, or the break cluster covering
    the foot spot all end up baked into the table geometry.
    """
    info = probe(path)
    last = info.frame_count if end_frame is None else min(end_frame, info.frame_count)
    if last <= start_frame:
        last = start_frame + 1

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames: List[np.ndarray] = []
    try:
        if info.frame_count > 0:
            targets = np.linspace(start_frame, max(start_frame, last - 1), count)
            targets = sorted({int(round(t)) for t in targets})
            for t in targets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
                ok, frame = cap.read()
                if ok:
                    frames.append(resize_to_width(frame, max_width))
        if len(frames) < max(3, count // 3):
            # Seeking is unreliable for some codecs; fall back to a sequential
            # pass with a stride.
            frames = []
            cap.release()
            cap = cv2.VideoCapture(str(path))
            stride = max(1, (last - start_frame) // max(1, count))
            idx = 0
            while len(frames) < count:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx >= start_frame and (idx - start_frame) % stride == 0:
                    frames.append(resize_to_width(frame, max_width))
                idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return frames


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


_FOURCC_BY_SUFFIX = {
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".avi": "MJPG",
    ".mkv": "mp4v",
    ".mov": "mp4v",
}


class VideoSink:
    """Lazily-opened writer that adopts the size of the first frame written."""

    def __init__(self, path: Union[str, Path], fps: float) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            fourcc_str = _FOURCC_BY_SUFFIX.get(self.path.suffix.lower(), "mp4v")
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Could not open video writer for {self.path} "
                    f"(codec {fourcc_str}). Try a different --output extension."
                )
        elif self._size != (w, h):
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class TrackCsvWriter:
    """Streaming per-frame track export.

    Streaming rather than accumulating: the old script grew two unbounded lists
    for the whole clip, which is both a memory leak and an O(n^2) slowdown as it
    rescanned them every frame.
    """

    COLUMNS = [
        "frame", "t_s", "track_id", "label", "ball_type", "state", "observed",
        "x_in", "y_in", "x_px", "y_px", "vx_in_s", "vy_in_s", "speed_in_s",
    ]

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.COLUMNS)
        self.rows_written = 0

    def write_frame(self, frame: int, t_s: float, tracks: Sequence[Any]) -> None:
        for track in tracks:
            pos = track.kf.position
            vel = track.kf.velocity
            img = track.last_image_xy
            self._writer.writerow(
                [
                    frame,
                    round(t_s, 4),
                    track.track_id,
                    track.label,
                    track.ball_type,
                    track.state.value,
                    int(track.time_since_update == 0),
                    round(float(pos[0]), 4),
                    round(float(pos[1]), 4),
                    round(float(img[0]), 2),
                    round(float(img[1]), 2),
                    round(float(vel[0]), 3),
                    round(float(vel[1]), 3),
                    round(float(track.speed), 3),
                ]
            )
            self.rows_written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TrackCsvWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")
