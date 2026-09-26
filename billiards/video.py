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
import os
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


def open_capture(path: Union[str, Path]) -> cv2.VideoCapture:
    """``cv2.VideoCapture`` for a file, whatever characters its path has.

    Some OpenCV builds on Windows cannot open a path with characters outside
    the system code page -- a match titled in Vietnamese, say.  Such a file is
    opened through its short (8.3) name instead, when Windows keeps one.
    """
    cap = cv2.VideoCapture(str(path))
    text = str(path)
    if cap.isOpened() or os.name != "nt" or text.isascii():
        return cap
    short = _short_path(text)
    if short and short != text:
        alt = cv2.VideoCapture(short)
        if alt.isOpened():
            cap.release()
            return alt
    return cap


def _short_path(path: str) -> Optional[str]:
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))  # type: ignore[attr-defined]
        return buf.value if 0 < n < len(buf) else None
    except Exception:
        return None


def probe(path: Union[str, Path]) -> VideoInfo:
    cap = open_capture(path)
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


#: Undecodable frames in a row that are skipped before a file is taken to end.
_MAX_BAD_FRAMES = 8


def read_frames(
    path: Union[str, Path],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_width: int = 0,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for the requested range.

    A frame that will not decode in the middle of a file -- a damaged
    download, a glitch in a recording -- is skipped rather than taken for the
    end, up to a few in a row, while the file says there is more to come.
    """
    cap = open_capture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        idx = start_frame
        failures = 0
        while True:
            if end_frame is not None and idx >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                if total > 0 and idx < total - 1 and failures < _MAX_BAD_FRAMES:
                    failures += 1
                    idx += 1
                    continue
                break
            failures = 0
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

    cap = open_capture(path)
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
            cap = open_capture(path)
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
    ".webm": "VP80",
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".avi": "MJPG",
    ".mkv": "mp4v",
    ".mov": "mp4v",
}


class VideoSink:
    """Lazily-opened writer that adopts the size of the first frame written."""

    def __init__(self, path: Union[str, Path], fps: float, fourcc: Optional[str] = None) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            fourcc_str = self.fourcc or _FOURCC_BY_SUFFIX.get(self.path.suffix.lower(), "mp4v")
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


class FfmpegSink:
    """H.264 through the ``ffmpeg`` that ships with ``imageio-ffmpeg``.

    OpenCV's own writers cannot make a file a browser will play on most
    installs: its FFmpeg is built without an H.264 encoder, Windows' Media
    Foundation one ignores the quality setting (~50 Mbit/s), and VP8 writes at
    ~40 fps.  This pipes raw frames to a real encoder instead, at ~180 fps.
    """

    def __init__(self, path: Union[str, Path], fps: float, crf: int = 20) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.crf = int(crf)
        self._proc: Optional[Any] = None
        self._size: Optional[Tuple[int, int]] = None
        self.frames_written = 0

    @staticmethod
    def available() -> bool:
        return _ffmpeg_exe() is not None

    def write(self, frame: np.ndarray) -> None:
        import subprocess

        h, w = frame.shape[:2]
        if self._proc is None:
            # yuv420p needs even dimensions.
            self._size = (w - w % 2, h - h % 2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                _ffmpeg_exe(), "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._size[0]}x{self._size[1]}", "-r", f"{self.fps:.6f}", "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if (w, h) != self._size:
            sw, sh = self._size
            if 0 <= w - sw <= 1 and 0 <= h - sh <= 1:
                frame = frame[:sh, :sw]  # the odd pixel yuv420p cannot take
            else:
                frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait()
            self._proc = None

    def __enter__(self) -> "FfmpegSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


_BROWSER_CODEC: Optional[Tuple[str, str]] = None


def browser_codec() -> Tuple[str, str]:
    """(writer, file suffix) for video a web browser can play, best first.

    ``ffmpeg`` (H.264 via imageio-ffmpeg), then OpenCV's ``avc1`` (H.264, on
    Windows through Media Foundation), then ``VP80`` (WebM), each tried once by
    writing and reading back a few frames.  ``mp4v`` if none work -- a browser
    will not play that, and the app shows such a video frame by frame instead.
    """
    global _BROWSER_CODEC
    if _BROWSER_CODEC is not None:
        return _BROWSER_CODEC
    if FfmpegSink.available():
        _BROWSER_CODEC = ("ffmpeg", ".mp4")
        return _BROWSER_CODEC
    import tempfile

    for fourcc, suffix, reads_as in (("avc1", ".mp4", "h264"), ("VP80", ".webm", "vp80")):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / f"probe{suffix}")
            try:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 25.0, (64, 48))
                ok = writer.isOpened()
                if ok:
                    for i in range(3):
                        writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
                writer.release()
                if not ok:
                    continue
                cap = cv2.VideoCapture(path)
                code = int(cap.get(cv2.CAP_PROP_FOURCC))
                cap.release()
                got = "".join(chr((code >> (8 * k)) & 0xFF) for k in range(4)).lower()
                if got == reads_as:
                    _BROWSER_CODEC = (fourcc, suffix)
                    return _BROWSER_CODEC
            except Exception:
                continue
    _BROWSER_CODEC = ("mp4v", ".mp4")
    return _BROWSER_CODEC


def open_sink(path: Union[str, Path], fps: float, writer: Optional[str] = None) -> Any:
    """A video sink for ``path``: ``writer`` is a fourcc, ``"ffmpeg"``, or None
    for whatever the suffix implies (``VideoSink``'s table)."""
    if writer == "ffmpeg":
        return FfmpegSink(path, fps)
    return VideoSink(path, fps, fourcc=writer)


class TrackCsvWriter:
    """Streaming per-frame track export.

    Streaming rather than accumulating: the old script grew two unbounded lists
    for the whole clip, which is both a memory leak and an O(n^2) slowdown as it
    rescanned them every frame.
    """

    COLUMNS = [
        "frame", "t_s", "track_id", "label", "number", "ball_type", "state", "observed",
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
                    "" if track.number is None else track.number,
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
