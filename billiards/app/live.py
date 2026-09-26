"""Tracking a table as it is played: a camera, a stream, or a file in real time.

The difference from a file is that frames do not wait.  A reader thread keeps
only the newest frame, and the tracker takes whichever that is when it is
ready for the next one, so when tracking is slower than the camera it skips
frames rather than falling ever further behind.  Each frame keeps the number
it had in the stream, so a skipped frame is a gap the tracker's clock sees,
exactly as a frame missed by a screen recorder is.

Before tracking the table has to be found, from frames spread over the first
few seconds rather than one (see ``table.calibrate``); until it is, the page
shows the camera's picture and says what it is waiting for.  If it is never
found -- no table in view, or not enough of it -- corners placed by hand on
the page are used instead.

A session can be recorded; it is then saved as a run like any other, and the
results page can replay it.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..pipeline import TrackingPipeline
from ..table import CalibrationResult, calibrate, estimate_cloth_color, table_from_corners
from ..video import TrackCsvWriter, browser_codec, open_capture, open_sink, resize_to_width, write_json
from .jobs import Preview, ball_record, build_viewer_data, encode_jpeg, table_record
from .workspace import Workspace, build_config


def describe_source(spec: Dict[str, Any], ws: Optional[Workspace] = None) -> str:
    kind = spec.get("kind")
    if kind == "camera":
        return f"camera {int(spec.get('index', 0))}"
    if kind == "url":
        return str(spec.get("url"))
    if kind == "file" and ws is not None:
        video = ws.video(str(spec.get("video_id")))
        return f"{video['name'] if video else spec.get('video_id')} (replayed live)"
    return str(spec)


def list_cameras(limit: int = 4) -> List[Dict[str, Any]]:
    """Cameras that open, by index.  Opening one takes up to a second."""
    found = []
    api = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.CAP_ANY
    for i in range(limit):
        cap = cv2.VideoCapture(i, api)
        try:
            if cap.isOpened():
                found.append({
                    "index": i,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                })
        finally:
            cap.release()
    return found


def _is_windows() -> bool:
    import os

    return os.name == "nt"


class FrameSource:
    """Keeps the newest frame of a stream, numbered as the stream counts."""

    def __init__(self, spec: Dict[str, Any], ws: Workspace, max_width: int) -> None:
        self.spec = spec
        self.ws = ws
        self.max_width = max_width
        self.fps = 30.0
        self.realtime_file = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Condition()
        self._latest: Optional[Tuple[int, np.ndarray]] = None
        self._stop = threading.Event()
        self.ended = False
        self.error: Optional[str] = None
        self.frames_read = 0
        self._t_first: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self.speed = float(spec.get("speed") or 1.0)

    def open(self) -> None:
        kind = self.spec.get("kind")
        if kind == "camera":
            index = int(self.spec.get("index", 0))
            api = cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                raise RuntimeError(f"camera {index} did not open (in use by another app?)")
        elif kind == "url":
            url = str(self.spec.get("url") or "").strip()
            if not url:
                raise ValueError("no stream address given")
            # Without a timeout a wrong address takes FFmpeg half a minute to
            # give up on, and a stream that stalls blocks the reader for good.
            params = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000]
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(url)
            if not cap.isOpened():
                cap = cv2.VideoCapture(url)  # another backend may know the scheme
            if not cap.isOpened():
                raise RuntimeError(f"could not open the stream at {url}")
        elif kind == "file":
            video = self.ws.video(str(self.spec.get("video_id")))
            if video is None:
                raise KeyError(self.spec.get("video_id"))
            cap = open_capture(video["path"])
            if not cap.isOpened():
                raise RuntimeError(f"could not open {video['name']}")
            self.realtime_file = True
            start = float(self.spec.get("start_s") or 0.0)
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        else:
            raise ValueError(f"unknown source kind {kind!r}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if np.isfinite(fps) and 1.0 < fps < 241.0 else 30.0
        self._cap = cap
        self._thread = threading.Thread(target=self._read_loop, name="live-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._cap is not None
        cap = self._cap
        t0 = time.perf_counter()
        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) if self.realtime_file else 0
        first_index = index
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.realtime_file:
                        break
                    failures += 1
                    if failures > 50:
                        self.error = "the stream stopped sending pictures"
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                if self.realtime_file:
                    # A file is read faster than it plays: hold each frame
                    # back until its moment comes.
                    due = t0 + (index - first_index) / (self.fps * self.speed)
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                frame = resize_to_width(frame, self.max_width)
                with self._lock:
                    self._latest = (index, frame)
                    self.frames_read += 1
                    if self._t_first is None:
                        self._t_first = time.perf_counter()
                    self._lock.notify_all()
                index += 1
        finally:
            with self._lock:
                self.ended = True
                self._lock.notify_all()
            cap.release()

    def measured_fps(self) -> float:
        if self._t_first is None or self.frames_read < 10:
            return self.fps
        return self.frames_read / max(time.perf_counter() - self._t_first, 1e-6)

    def next(self, after: int, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """The newest frame numbered after ``after``, or None at the end."""
        deadline = time.perf_counter() + timeout
        with self._lock:
            while True:
                if self._latest is not None and self._latest[0] > after:
                    return self._latest
                if self.ended:
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(out, text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return out


class LiveSession:
    def __init__(self, ws: Workspace, source: Dict[str, Any], settings: Dict[str, Any],
                 record: bool = False) -> None:
        self.ws = ws
        self.source_spec = source
        self.settings = settings
        self.record = record
        self.status = "connecting"
        self.message = "opening " + describe_source(source, ws)
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.preview = Preview()
        self.snapshot: Optional[bytes] = None  # the raw picture, for placing corners
        self.image_size: Optional[List[int]] = None
        self.started = time.time()
        self.finished: Optional[float] = None
        self.frames_processed = 0
        self.frames_skipped = 0
        self.fps_in = 0.0
        self.fps_out = 0.0
        self.balls: List[Dict[str, Any]] = []
        self.events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.event_counts: Dict[str, int] = {}
        self.shot_line: Optional[str] = None
        self.shots: List[Dict[str, Any]] = []
        self.run_id: Optional[str] = None
        self.table: Optional[Dict[str, Any]] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name="live-session", daemon=True)
        self._pipeline: Optional[TrackingPipeline] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "source": self.source_spec,
            "source_name": describe_source(self.source_spec, self.ws),
            "settings": self.settings,
            "record": self.record,
            "running": self.running,
            "started": self.started,
            "finished": self.finished,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "fps_in": round(self.fps_in, 1),
            "fps_out": round(self.fps_out, 1),
            "balls": self.balls,
            "recent_events": list(self.events)[-40:],
            "event_counts": dict(self.event_counts),
            "shot": self.shot_line,
            "shots": self.shots,
            "run_id": self.run_id,
            "image_size": self.image_size,
            "table": self.table,
            "ball_set": None if self._pipeline is None else self._pipeline.tracker.ball_set,
        }

    # -- the session ---------------------------------------------------------

    def _main(self) -> None:
        cfg = build_config(self.settings)
        source = FrameSource(self.source_spec, self.ws, cfg.max_frame_width)
        try:
            source.open()
            calib = self._find_table(source, cfg)
            if calib is None:
                return
            self._track(source, cfg, calib)
        except Exception as exc:
            self.status = "error"
            self.error = str(exc) or exc.__class__.__name__
            self.message = self.error
            self.traceback = traceback.format_exc()
        finally:
            source.close()
            self.finished = time.time()
            self.preview.close()

    def _publish_raw(self, frame: np.ndarray, text: str) -> None:
        self.snapshot = encode_jpeg(frame, max_width=0, quality=85)
        self.image_size = [int(frame.shape[1]), int(frame.shape[0])]
        self.preview.publish(encode_jpeg(_caption(frame, text)))

    def _find_table(self, source: FrameSource, cfg: Any) -> Optional[CalibrationResult]:
        """Frames spread over the first seconds, then the calibration."""
        self.status = "calibrating"
        want = max(6, min(cfg.table.calibration_frames, 12))
        spacing_s = 0.12
        while not self._stop.is_set():
            frames: List[np.ndarray] = []
            last_idx, last_take = -1, 0.0
            self.message = "looking for the table"
            while len(frames) < want and not self._stop.is_set():
                got = source.next(last_idx, timeout=2.0)
                if got is None:
                    if source.ended:
                        self.status = "ended"
                        self.message = source.error or "the source ended before the table was found"
                        return None
                    continue
                last_idx, frame = got
                now = time.perf_counter()
                if now - last_take >= spacing_s / max(source.speed, 1e-6) or not frames:
                    frames.append(frame)
                    last_take = now
                self._publish_raw(frame, f"Looking for the table... {len(frames)}/{want}")
            if self._stop.is_set():
                break
            try:
                corners = self.settings.get("corners")
                if corners is not None:
                    h, w = frames[0].shape[:2]
                    table = table_from_corners(corners, cfg, image_size=(w, h))
                    bed = table.bed_mask((h, w), margin_in=2.0 * table.ball_diameter_in)
                    cloth = estimate_cloth_color([cv2.bitwise_and(f, f, mask=bed) for f in frames], cfg,
                                                 neutral_ok=True)
                    return CalibrationResult(table=table, cloth=cloth, frames_used=len(frames),
                                             frames_attempted=len(frames), corner_spread_px=0.0)
                return calibrate(frames, cfg)
            except (RuntimeError, ValueError):
                self.status = "no_table"
                self.message = ("No table found in the picture yet. Trying again; "
                                "or place the corners by hand.")
                self._publish_raw(frames[-1], "No table found - trying again")
                self._stop.wait(1.5)
        self.status = "stopped"
        self.message = "stopped"
        return None

    def _track(self, source: FrameSource, cfg: Any, calib: CalibrationResult) -> None:
        fps = source.fps if source.realtime_file else source.measured_fps()
        pipeline = TrackingPipeline(cfg, calib.table, calib.cloth, fps)
        self._pipeline = pipeline
        self.table = table_record(calib.table, calib.cloth)
        self.status = "tracking"
        self.message = "tracking"

        sink = csv_writer = None
        folder: Optional[Path] = None
        video_name = None
        if self.record:
            vid = str(self.source_spec.get("video_id") or "live")
            self.run_id, folder = self.ws.new_run_dir(vid if len(vid) >= 6 else "live00")
            writer, suffix = browser_codec()
            video_name = f"tracked{suffix}"
            sink = open_sink(folder / video_name, fps, writer)
            csv_writer = TrackCsvWriter(folder / "tracks.csv")
            self._save_meta(folder, "running", video_name, writer != "mp4v")

        balls_seen: Dict[int, Dict[str, Any]] = {}
        last_idx = -1
        first_idx: Optional[int] = None
        last_annotated: Optional[np.ndarray] = None
        t_start = time.perf_counter()
        last_ui = last_snap = 0.0
        try:
            while not self._stop.is_set():
                got = source.next(last_idx, timeout=2.0)
                if got is None:
                    if source.ended:
                        break
                    continue
                idx, frame = got
                if first_idx is None:
                    first_idx = idx
                if last_idx >= 0 and idx > last_idx + 1:
                    self.frames_skipped += idx - last_idx - 1
                    # Keep the recording on the stream's clock: the frames
                    # the tracker had no time for repeat the last picture.
                    if sink is not None and last_annotated is not None:
                        for _ in range(min(idx - last_idx - 1, int(fps) * 2)):
                            sink.write(last_annotated)
                last_idx = idx
                result = pipeline.process(frame, idx - first_idx, annotate=True)
                self.frames_processed += 1
                annotated = result.annotated if result.annotated is not None else frame
                last_annotated = annotated
                if sink is not None:
                    sink.write(annotated)
                if csv_writer is not None:
                    csv_writer.write_frame(idx - first_idx, result.t_s, result.tracks)
                for track in result.tracks:
                    balls_seen[track.track_id] = ball_record(track)
                for e in result.events:
                    d = e.to_dict()
                    d["labels"] = [pipeline.label_of(t) for t in e.track_ids]
                    self.events.append(d)
                    self.event_counts[d["type"]] = self.event_counts.get(d["type"], 0) + 1
                now = time.perf_counter()
                elapsed = max(now - t_start, 1e-6)
                self.fps_out = self.frames_processed / elapsed
                self.fps_in = source.measured_fps()
                if now - last_snap >= 1.0:
                    last_snap = now
                    self.snapshot = encode_jpeg(frame, max_width=0, quality=85)
                if now - last_ui >= 0.05:
                    last_ui = now
                    self.preview.publish(encode_jpeg(annotated))
                    self.balls = [
                        {**ball_record(t), "x": round(float(t.kf.position[0]), 1),
                         "y": round(float(t.kf.position[1]), 1), "speed": round(float(t.speed), 1),
                         "state": t.state.value}
                        for t in result.tracks
                    ]
                    self.shot_line = pipeline.shots.live_description()
                    self.shots = pipeline.shots.to_list()
                    self.message = ("tracking" if pipeline.view_valid
                                    else "the table is out of view - waiting for it")
            self.status = "ended" if source.ended and not self._stop.is_set() else "stopped"
            self.message = source.error or ("the source ended" if self.status == "ended" else "stopped")
        finally:
            pipeline.finish()
            self.shots = pipeline.shots.to_list()
            if sink is not None:
                sink.close()
            if csv_writer is not None:
                csv_writer.close()
            if folder is not None:
                self._write_results(folder, pipeline, calib, fps, balls_seen, video_name,
                                    last_idx - (first_idx or 0) + 1, time.perf_counter() - t_start)

    def _write_results(self, folder: Path, pipeline: TrackingPipeline, calib: CalibrationResult,
                       fps: float, balls: Dict[int, Dict[str, Any]], video_name: Optional[str],
                       frames: int, wall_s: float) -> None:
        h, w = (self.image_size[1], self.image_size[0]) if self.image_size else (0, 0)
        summary = {
            "video": {"path": describe_source(self.source_spec, self.ws), "width": w, "height": h,
                      "fps": round(fps, 3), "frame_count": frames, "duration_s": round(frames / fps, 2)},
            "calibration": calib.to_dict(),
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "live": True,
            "wall_seconds": round(wall_s, 2),
            "processing_fps": round(self.frames_processed / max(wall_s, 1e-6), 2),
            **pipeline.summary(),
            "shot_log": pipeline.shots.to_list(),
            "event_log": [e.to_dict() for e in pipeline.all_events],
        }
        write_json(folder / "run.json", summary)
        viewer = build_viewer_data(folder / "tracks.csv", summary, balls,
                                   table_record(pipeline.table, pipeline.cloth))
        viewer["start_frame"] = 0
        write_json(folder / "viewer.json", viewer)
        if self.preview.jpeg:
            (folder / "thumb.jpg").write_bytes(self.preview.jpeg)
        writer, _ = browser_codec()
        self._save_meta(folder, "done", video_name, writer != "mp4v", summary, len(viewer["tracks"]))

    def _save_meta(self, folder: Path, status: str, video_name: Optional[str], playable: bool,
                   summary: Optional[Dict[str, Any]] = None, n_tracks: int = 0) -> None:
        name = describe_source(self.source_spec, self.ws)
        meta = {
            "id": folder.name,
            "video_id": self.source_spec.get("video_id") or "live",
            "video_name": f"Live: {name}",
            "status": status,
            "message": "recorded live" if status == "done" else "recording",
            "error": None,
            "created": self.started,
            "started": self.started,
            "finished": time.time() if status == "done" else None,
            "progress": 1.0 if status == "done" else 0.0,
            "settings": self.settings,
            "video_file": video_name,
            "browser_playable": playable,
            "live": True,
            "brief": None if summary is None else {
                "tracks": n_tracks,
                "shots": [s.get("summary") for s in summary.get("shot_log", [])],
                "events": summary.get("events", {}),
                "clock": summary.get("clock"),
                "processing_fps": summary.get("processing_fps"),
                "frames": summary.get("frames_processed"),
            },
        }
        self.ws.save_run_meta(folder.name, meta)


class LiveManager:
    """At most one live session at a time."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.session: Optional[LiveSession] = None
        self._lock = threading.Lock()

    def start(self, source: Dict[str, Any], settings: Dict[str, Any], record: bool) -> LiveSession:
        with self._lock:
            if self.session is not None and self.session.running:
                self.session.stop()
                self.session.join(10.0)
            self.session = LiveSession(self.ws, source, settings, record)
            self.session.start()
            return self.session

    def stop(self) -> None:
        with self._lock:
            if self.session is not None:
                self.session.stop()
                self.session.join(15.0)

    def state(self) -> Dict[str, Any]:
        if self.session is None:
            return {"status": "idle", "running": False, "message": "not started"}
        return self.session.to_dict()
