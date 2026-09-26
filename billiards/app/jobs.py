"""Tracking runs in the background, for the app.

A run is the command-line ``track`` of one video with its saved settings,
written into its own folder under the workspace, plus what the browser needs
to follow it: a progress figure, the latest annotated frame as a JPEG for the
live preview, the events so far, and at the end ``viewer.json`` -- every
ball's path in a form the results page can draw without parsing a CSV.
"""

from __future__ import annotations

import csv
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..pipeline import FrameResult, RunOptions, TrackingPipeline, run
from ..render import overhead_flips
from ..video import browser_codec, probe
from .workspace import Workspace, build_config, write_json


def encode_jpeg(frame: np.ndarray, max_width: int = 960, quality: int = 78) -> bytes:
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        frame = cv2.resize(frame, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class Preview:
    """The latest picture of something running, for any number of watchers."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.seq = 0
        self.closed = False

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self.jpeg = jpeg
            self.seq += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 5.0) -> Tuple[int, Optional[bytes]]:
        """The next picture after ``last_seq`` (or the current one on timeout)."""
        with self._cond:
            if self.seq == last_seq and not self.closed:
                self._cond.wait(timeout)
            return self.seq, self.jpeg


def ball_record(track: Any) -> Dict[str, Any]:
    """What the pages show about a ball, from a live track."""
    b, g, r = track.signature.bgr
    return {
        "id": track.track_id,
        "label": track.label,
        "number": track.number,
        "type": track.ball_type,
        "colour": "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)),
    }


def table_record(table: Any, cloth: Any = None) -> Dict[str, Any]:
    flip_x, flip_y = overhead_flips(table)
    out = {
        "length_in": table.length_in,
        "width_in": table.width_in,
        "ball_diameter_in": table.ball_diameter_in,
        "has_pockets": table.has_pockets,
        "pockets": [[round(float(x), 2), round(float(y), 2)] for x, y in table.pockets_table()],
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
        "corners_image": [[round(float(x), 1), round(float(y), 1)] for x, y in table.corners_image],
    }
    if cloth is not None:
        hsv = np.uint8([[[int(cloth.hue) % 180, int(np.clip(cloth.sat, 0, 255)),
                          int(np.clip(cloth.val * 0.8, 0, 255))]]])
        b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out["cloth_colour"] = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    return out


def build_viewer_data(
    csv_path: Path,
    summary: Dict[str, Any],
    balls: Dict[int, Dict[str, Any]],
    table: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Every ball's path, arranged for the results page."""
    tracks: Dict[int, Dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                tid = int(row["track_id"])
                t = tracks.get(tid)
                if t is None:
                    t = tracks[tid] = {
                        "id": tid, "f": [], "x": [], "y": [], "v": [], "o": [], "labels": [],
                    }
                frame = int(row["frame"])
                t["f"].append(frame)
                t["x"].append(round(float(row["x_in"]), 2))
                t["y"].append(round(float(row["y_in"]), 2))
                t["v"].append(round(float(row["speed_in_s"]), 1))
                t["o"].append(int(row["observed"]))
                if not t["labels"] or t["labels"][-1][1] != row["label"]:
                    t["labels"].append([frame, row["label"]])
                t["type"] = row["ball_type"]
                t["number"] = int(row["number"]) if row.get("number") else None
    out_tracks = []
    for tid, t in sorted(tracks.items()):
        xs, ys, obs = np.array(t["x"]), np.array(t["y"]), np.array(t["o"])
        steps = np.hypot(np.diff(xs), np.diff(ys)) if len(xs) > 1 else np.zeros(0)
        info = balls.get(tid, {})
        out_tracks.append({
            **t,
            "label": t["labels"][-1][1] if t["labels"] else f"#{tid}",
            "colour": info.get("colour", "#9aa0a6"),
            "first": t["f"][0],
            "last": t["f"][-1],
            "observed_frames": int(obs.sum()),
            "distance_in": round(float(steps.sum()), 1),
            "top_speed_in_s": round(float(max(t["v"])) if t["v"] else 0.0, 1),
        })
    video = summary.get("video", {})
    return {
        "fps": video.get("fps"),
        "frame_count": video.get("frame_count"),
        "width": video.get("width"),
        "height": video.get("height"),
        "table": table,
        "tracks": out_tracks,
        "events": summary.get("event_log", []),
        "shots": summary.get("shot_log", []),
        "clock": summary.get("clock"),
        "ball_set": summary.get("ball_set"),
    }


class RunJob:
    def __init__(self, rid: str, folder: Path, video: Dict[str, Any], settings: Dict[str, Any]) -> None:
        self.id = rid
        self.folder = folder
        self.video = video
        self.settings = settings
        self.status = "queued"
        self.message = "waiting to start"
        self.error: Optional[str] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.frame = 0
        self.frames_done = 0
        self.total_frames = 0
        self.fps = 0.0
        self.events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.event_counts: Dict[str, int] = {}
        self.shot_line: Optional[str] = None
        self.balls_live: List[Dict[str, Any]] = []
        self.preview = Preview()
        self.cancel_event = threading.Event()
        self._balls: Dict[int, Dict[str, Any]] = {}
        self._table: Any = None
        self._cloth: Any = None
        self._last_jpeg = 0.0
        self._t0 = 0.0
        self.video_file: Optional[str] = None
        self.browser_playable = False

    # -- the loop's callbacks ----------------------------------------------

    def on_progress(self, msg: str) -> None:
        if msg.startswith("calibrated"):
            self.status = "running"
            self.message = msg

    def on_frame(self, result: FrameResult, pipeline: TrackingPipeline) -> None:
        self.frame = result.frame_index
        self.frames_done += 1
        if self.frames_done == 1:
            self._t0 = time.time()  # the rate of tracking, not of calibrating
        self._table, self._cloth = pipeline.table, pipeline.cloth
        elapsed = max(time.time() - self._t0, 1e-6)
        self.fps = self.frames_done / elapsed
        for track in result.tracks:
            self._balls[track.track_id] = ball_record(track)
        for e in result.events:
            d = e.to_dict()
            d["labels"] = [pipeline.label_of(t) for t in e.track_ids]
            self.events.append(d)
            self.event_counts[d["type"]] = self.event_counts.get(d["type"], 0) + 1
        self.shot_line = pipeline.shots.live_description()
        now = time.time()
        if result.annotated is not None and now - self._last_jpeg >= 0.08:
            self._last_jpeg = now
            self.preview.publish(encode_jpeg(result.annotated))
            self.balls_live = [
                {**ball_record(t), "x": round(float(t.kf.position[0]), 1),
                 "y": round(float(t.kf.position[1]), 1), "speed": round(float(t.speed), 1),
                 "state": t.state.value}
                for t in result.tracks
            ]

    # -- reporting -----------------------------------------------------------

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.total_frames:
            return 0.0
        return float(np.clip(self.frames_done / self.total_frames, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        eta = None
        if self.status == "running" and self.fps > 0 and self.total_frames:
            eta = max(0.0, (self.total_frames - self.frames_done) / self.fps)
        return {
            "id": self.id,
            "video_id": self.video["id"],
            "video_name": self.video["name"],
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": round(self.progress, 4),
            "frame": self.frame,
            "frames_done": self.frames_done,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 1),
            "eta_s": None if eta is None else round(eta, 1),
            "event_counts": dict(self.event_counts),
            "recent_events": list(self.events)[-30:],
            "shot": self.shot_line,
            "balls": self.balls_live,
            "settings": self.settings,
            "video_file": self.video_file,
            "browser_playable": self.browser_playable,
            "live": False,
        }

    # -- the run -------------------------------------------------------------

    def execute(self, ws: Workspace) -> None:
        self.started = self._t0 = time.time()
        self.status = "calibrating"
        self.message = "finding the table"
        self._save_meta(ws)
        try:
            cfg = build_config(self.settings)
            info = probe(self.video["path"])
            start = int(round(self.settings["start_s"] * info.fps))
            end = None if self.settings["end_s"] is None else int(round(self.settings["end_s"] * info.fps))
            last = info.frame_count if end is None else min(end, info.frame_count)
            self.total_frames = max(0, last - start)
            writer, suffix = browser_codec()
            self.browser_playable = writer != "mp4v"
            video_out = self.folder / f"tracked{suffix}"
            self.video_file = video_out.name
            summary = run(cfg, RunOptions(
                video=self.video["path"],
                output=str(video_out),
                export_csv=str(self.folder / "tracks.csv"),
                export_json=str(self.folder / "run.json"),
                start_frame=start,
                end_frame=end,
                table_corners=self.settings["corners"],
                progress_every=0,
                on_progress=self.on_progress,
                writer=writer,
                on_frame=self.on_frame,
                should_stop=self.cancel_event.is_set,
                annotate=True,
            ))
            self.status = "finishing"
            self.message = "writing the results"
            self._save_meta(ws)
            table_info = summary.get("table")
            viewer = build_viewer_data(
                self.folder / "tracks.csv", summary, self._balls,
                None if self._table is None else table_record(self._table, self._cloth),
            )
            viewer["start_frame"] = start
            write_json(self.folder / "viewer.json", viewer)
            self._write_thumb()
            self.status = "cancelled" if summary.get("stopped_early") else "done"
            self.message = (
                f"stopped after {summary['frames_processed']} frames"
                if summary.get("stopped_early")
                else f"{summary['frames_processed']} frames in {summary['wall_seconds']} s"
            )
            self.brief = {
                "tracks": len(viewer["tracks"]),
                "shots": [s.get("summary") for s in summary.get("shot_log", [])],
                "events": summary.get("events", {}),
                "clock": summary.get("clock"),
                "processing_fps": summary.get("processing_fps"),
                "frames": summary.get("frames_processed"),
                "table_info": table_info,
            }
        except Exception as exc:  # reported to the page, not raised
            from .preview import for_the_app

            self.status = "failed"
            self.error = for_the_app(str(exc)) or exc.__class__.__name__
            self.message = self.error
            (self.folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            self.finished = time.time()
            self.preview.close()
            self._save_meta(ws)

    def _write_thumb(self) -> None:
        if self.preview.jpeg:
            (self.folder / "thumb.jpg").write_bytes(self.preview.jpeg)

    def _save_meta(self, ws: Workspace) -> None:
        meta = self.to_dict()
        meta.pop("recent_events", None)
        meta.pop("balls", None)
        meta["brief"] = getattr(self, "brief", None)
        ws.save_run_meta(self.id, meta)


class RunManager:
    """Runs tracking jobs one (or ``parallel``) at a time, in the background."""

    def __init__(self, ws: Workspace, parallel: int = 1) -> None:
        self.ws = ws
        self._jobs: Dict[str, RunJob] = {}
        self._queue: Deque[RunJob] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._workers = [
            threading.Thread(target=self._work, name=f"run-worker-{i}", daemon=True)
            for i in range(max(1, parallel))
        ]
        for w in self._workers:
            w.start()

    def submit(self, vid: str, overrides: Optional[Dict[str, Any]] = None) -> RunJob:
        video = self.ws.video(vid)
        if video is None:
            raise KeyError(vid)
        if video.get("error"):
            raise ValueError(f"{video['name']}: {video['error']}")
        settings = self.ws.settings(vid)
        if overrides:
            settings = self.ws.save_settings(vid, overrides)
        rid, folder = self.ws.new_run_dir(vid)
        job = RunJob(rid, folder, video, settings)
        with self._wake:
            self._jobs[rid] = job
            self._queue.append(job)
            job._save_meta(self.ws)
            self._wake.notify()
        return job

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                job = self._queue.popleft()
            if job.cancel_event.is_set():
                job.status, job.message = "cancelled", "cancelled before it started"
                job.finished = time.time()
                job.preview.close()
                job._save_meta(self.ws)
                continue
            job.execute(self.ws)

    def job(self, rid: str) -> Optional[RunJob]:
        return self._jobs.get(rid)

    def active(self) -> List[RunJob]:
        return [j for j in self._jobs.values() if j.status in ("queued", "calibrating", "running", "finishing")]

    def cancel(self, rid: str) -> bool:
        job = self._jobs.get(rid)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def describe(self, rid: str) -> Optional[Dict[str, Any]]:
        """A run's state: live if it is running here, else from its folder."""
        job = self._jobs.get(rid)
        if job is not None and job.status in ("queued", "calibrating", "running", "finishing"):
            return job.to_dict()
        meta = self.ws.run_meta(rid)
        if meta is None:
            return None
        # A run the app was closed in the middle of never finished.
        if meta.get("status") in ("queued", "calibrating", "running", "finishing") and job is None:
            meta["status"] = "failed"
            meta["error"] = meta["message"] = "the app was closed while this was running"
        return meta

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for meta in self.ws.runs():
            rid = meta.get("id")
            live = self.describe(rid) if rid else None
            out.append(live or meta)
        return out
