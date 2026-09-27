"""Where the app keeps things: the footage library, per-video settings, runs.

Everything lives under one folder (``billiards-workspace/`` by default)::

    library.json          folders to scan and single files added by hand
    settings/<video>.json how to track each video (table size, corners, ...)
    runs/<run>/           one folder per run: meta.json, run.json, tracks.csv,
                          the annotated video and viewer.json for the browser
    uploads/              videos dropped into the browser
    downloads/            parts of videos fetched from a link (YouTube, ...)
    cache/                probed video metadata and thumbnails

A video is known by a short hash of its absolute path, so the same file keeps
its settings and runs however it was added.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional

import cv2
import numpy as np

from ..config import TABLE_PRESETS, Config, TableConfig
from ..video import open_capture, probe, resize_to_width

VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv",
    ".ts", ".mts", ".m2ts", ".flv", ".3gp",
}

#: How a video is tracked unless its settings say otherwise.
DEFAULT_SETTINGS: Dict[str, Any] = {
    "preset": "pool-9ft",
    "ball_set": "auto",
    "numbers": "1-15",
    "start_s": 0.0,
    "end_s": None,
    #: Four table corners in processing-resolution pixels, or None to find
    #: the table automatically.
    "corners": None,
    "max_width": 1280,
    #: Burn the top-down diagram and status text into the video.  The app
    #: draws its own, so its videos are the picture alone by default.
    "burn_in_panel": False,
    "draw_trails": True,
    #: Also look for balls against the far cushion (``detector.search_raised_bed``).
    #: Off by default: it finds more, and on albin_fedor it finds a pot the
    #: default misses, but it also made a phantom next to a black ball there.
    "far_cushion": False,
}


def video_id(path: Path) -> str:
    key = os.path.normcase(str(Path(path).resolve()))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def clean_settings(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Settings with unknown keys dropped and every value checked."""
    out = dict(DEFAULT_SETTINGS)
    for key, value in (raw or {}).items():
        if key in DEFAULT_SETTINGS:
            out[key] = value
    if out["preset"] not in TABLE_PRESETS:
        raise ValueError(f"unknown table preset {out['preset']!r}; one of {sorted(TABLE_PRESETS)}")
    if out["ball_set"] not in ("auto", "standard", "tv"):
        raise ValueError("ball set must be auto, standard or tv")
    from ..balls import parse_numbers

    try:
        numbers = parse_numbers(out["numbers"])
    except ValueError:
        raise ValueError(f"numbers must look like 1-15 or 1-7,9, not {out['numbers']!r}")
    if not numbers or min(numbers) < 1 or max(numbers) > 15:
        raise ValueError("numbers must be between 1 and 15")
    out["start_s"] = max(0.0, float(out["start_s"] or 0.0))
    out["end_s"] = None if out["end_s"] in (None, "") else float(out["end_s"])
    if out["end_s"] is not None and out["end_s"] <= out["start_s"]:
        raise ValueError("the end time must be after the start time")
    if out["corners"] is not None:
        pts = np.asarray(out["corners"], dtype=np.float64)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            raise ValueError("corners must be four [x, y] points")
        out["corners"] = [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
    out["max_width"] = int(out["max_width"] or 0)
    if out["max_width"] and out["max_width"] < 320:
        raise ValueError("the processing width must be at least 320 px (or 0 for full size)")
    out["burn_in_panel"] = bool(out["burn_in_panel"])
    out["draw_trails"] = bool(out["draw_trails"])
    out["far_cushion"] = bool(out["far_cushion"])
    return out


def build_config(settings: Dict[str, Any], base: Optional[Config] = None) -> Config:
    """The tracker's configuration for these settings."""
    cfg = base if base is not None else Config()
    cfg.table.preset = settings["preset"]
    defaults = TableConfig()
    cfg.table.length_in = defaults.length_in
    cfg.table.width_in = defaults.width_in
    cfg.table.ball_diameter_in = defaults.ball_diameter_in
    cfg.apply_preset()
    cfg.balls.ball_set = settings["ball_set"]
    cfg.balls.numbers = str(settings["numbers"])
    cfg.max_frame_width = int(settings["max_width"])
    cfg.render.overhead_panel = bool(settings["burn_in_panel"])
    cfg.render.draw_hud = bool(settings["burn_in_panel"])
    cfg.render.draw_trajectories = bool(settings["draw_trails"])
    cfg.detector.search_raised_bed = bool(settings.get("far_cushion", False))
    return cfg


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def safe_filename(name: str) -> str:
    """A file name that is safe on every platform, keeping its suffix."""
    name = Path(name).name
    stem, suffix = os.path.splitext(name)
    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip(" .") or "video"
    suffix = re.sub(r"[^\w.]+", "", suffix)[:8]
    return f"{stem[:80]}{suffix}"


class Workspace:
    def __init__(self, root: Path, folders: Iterable[Path] = ()) -> None:
        self.root = Path(root).resolve()
        for sub in ("settings", "runs", "uploads", "downloads", "cache/thumbs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._library_path = self.root / "library.json"
        self._library = _read_json(self._library_path, {})
        self._library.setdefault("folders", [])
        self._library.setdefault("files", [])
        self._library.setdefault("hidden", [])
        self._meta_path = self.root / "cache" / "meta.json"
        self._meta: Dict[str, Any] = _read_json(self._meta_path, {})
        self._meta_dirty = False
        for own in self.own_folders():
            if str(own) not in self._library["folders"]:
                self._library["folders"].append(str(own))
        for folder in folders:
            self.add_folder(folder, save=False)
        self._save_library()

    # -- library -----------------------------------------------------------

    def _save_library(self) -> None:
        with self._lock:
            _write_json(self._library_path, self._library)

    def own_folders(self) -> List[Path]:
        """Folders whose videos the app put there, and may delete."""
        return [self.root / "uploads", self.root / "downloads"]

    def folders(self) -> List[str]:
        return list(self._library["folders"])

    def add_folder(self, folder: Path, save: bool = True) -> str:
        path = Path(folder).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"not a folder: {path}")
        with self._lock:
            if str(path) not in self._library["folders"]:
                self._library["folders"].append(str(path))
            if save:
                self._save_library()
        return str(path)

    def remove_folder(self, folder: str) -> None:
        with self._lock:
            self._library["folders"] = [f for f in self._library["folders"] if f != folder]
            self._save_library()

    def add_file(self, file: Path) -> Dict[str, Any]:
        path = Path(file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"not a file: {path}")
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"not a video file this app knows: {path.name}")
        with self._lock:
            if str(path) not in self._library["files"]:
                self._library["files"].append(str(path))
            vid = video_id(path)
            if vid in self._library["hidden"]:
                self._library["hidden"].remove(vid)
            self._save_library()
        found = self.video(vid)
        if found is None or found.get("error"):
            with self._lock:
                self._library["files"] = [f for f in self._library["files"] if f != str(path)]
                self._save_library()
            reason = found.get("error") if found else "could not be read"
            raise ValueError(f"{path.name}: {reason}")
        return found

    def forget(self, vid: str) -> None:
        """Take a video off the library list (the file itself is left alone,
        unless the app put it in the workspace: an upload or a download)."""
        video = self.video(vid)
        with self._lock:
            if video is not None:
                self._library["files"] = [f for f in self._library["files"] if f != video["path"]]
                if Path(video["path"]).parent in self.own_folders():
                    try:
                        Path(video["path"]).unlink()
                    except OSError:
                        pass
            if vid not in self._library["hidden"]:
                self._library["hidden"].append(vid)
            self._save_library()

    def _paths(self) -> List[Path]:
        seen, out = set(), []
        for folder in self._library["folders"]:
            try:
                entries = sorted(Path(folder).iterdir())
            except OSError:
                continue
            for p in entries:
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and p not in seen:
                    seen.add(p)
                    out.append(p)
        for f in self._library["files"]:
            p = Path(f)
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def videos(self) -> List[Dict[str, Any]]:
        hidden = set(self._library["hidden"])
        out = []
        for path in self._paths():
            vid = video_id(path)
            if vid in hidden:
                continue
            meta = self._probe(path)
            if meta is not None:
                out.append(meta)
        self._save_meta()
        return out

    def video(self, vid: str) -> Optional[Dict[str, Any]]:
        for path in self._paths():
            if video_id(path) == vid:
                meta = self._probe(path)
                self._save_meta()
                return meta
        return None

    def _probe(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            st = path.stat()
        except OSError:
            return None
        vid = video_id(path)
        key = f"{st.st_size}:{int(st.st_mtime)}"
        cached = self._meta.get(vid)
        if cached and cached.get("key") == key:
            meta = dict(cached)
        else:
            try:
                info = probe(path)
            except (FileNotFoundError, RuntimeError, cv2.error):
                meta = {"key": key, "error": "could not be opened as a video"}
            else:
                if info.width <= 0 or info.height <= 0:
                    meta = {"key": key, "error": "no picture in this file"}
                else:
                    meta = {"key": key, **info.to_dict()}
            with self._lock:
                self._meta[vid] = meta
                self._meta_dirty = True
            meta = dict(meta)
        meta.update({
            "id": vid,
            "name": path.name,
            "path": str(path),
            "folder": str(path.parent),
            "size_bytes": st.st_size,
            "modified": int(st.st_mtime),
        })
        meta.pop("key", None)
        return meta

    def _save_meta(self) -> None:
        with self._lock:
            if self._meta_dirty:
                _write_json(self._meta_path, self._meta)
                self._meta_dirty = False

    def save_upload(self, name: str, stream: BinaryIO, length: int) -> Dict[str, Any]:
        """Copy an uploaded video into ``uploads/`` and add it."""
        filename = safe_filename(name)
        if Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"{name}: not a video file this app knows")
        dest = self.root / "uploads" / filename
        n = 1
        while dest.exists():
            dest = dest.with_name(f"{Path(filename).stem} ({n}){Path(filename).suffix}")
            n += 1
        tmp = dest.with_name(dest.name + ".part")
        remaining = int(length)
        with tmp.open("wb") as fh:
            while remaining > 0:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            tmp.unlink(missing_ok=True)
            raise ValueError("the upload was cut short")
        os.replace(tmp, dest)
        found = self.video(video_id(dest))
        if found is None or found.get("error"):
            raise ValueError(f"{name} was saved, but it cannot be read as a video")
        return found

    # -- pictures ----------------------------------------------------------

    def read_frame(self, vid: str, t_s: Optional[float] = None, frame: Optional[int] = None,
                   max_width: int = 0) -> np.ndarray:
        video = self.video(vid)
        if video is None or video.get("error"):
            raise KeyError(vid)
        cap = open_capture(video["path"])
        try:
            fps = float(video.get("fps") or 30.0)
            total = int(video.get("frame_count") or 0)
            index = frame if frame is not None else int(round((t_s or 0.0) * fps))
            if total > 0:
                index = int(np.clip(index, 0, total - 1))
            if index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ok, img = cap.read()
            if not ok and index > 0:
                # Seeking is unreliable for some codecs: read up to it instead.
                cap.release()
                cap = open_capture(video["path"])
                for _ in range(index + 1):
                    ok, got = cap.read()
                    if not ok:
                        break
                    img = got
                ok = img is not None
            if not ok or img is None:
                raise RuntimeError(f"could not read frame {index} of {video['name']}")
            return resize_to_width(img, max_width)
        finally:
            cap.release()

    def thumbnail(self, vid: str) -> Path:
        path = self.root / "cache" / "thumbs" / f"{vid}.jpg"
        if not path.exists():
            video = self.video(vid)
            if video is None or video.get("error"):
                raise KeyError(vid)
            t = 0.15 * float(video.get("duration_s") or 0.0)
            img = self.read_frame(vid, t_s=t, max_width=480)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                path.write_bytes(buf.tobytes())
        return path

    # -- per-video settings ------------------------------------------------

    def settings(self, vid: str) -> Dict[str, Any]:
        saved = _read_json(self.root / "settings" / f"{vid}.json", {})
        try:
            return clean_settings(saved)
        except ValueError:
            return dict(DEFAULT_SETTINGS)

    def save_settings(self, vid: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.settings(vid), **(raw or {})}
        settings = clean_settings(merged)
        _write_json(self.root / "settings" / f"{vid}.json", settings)
        return settings

    # -- runs --------------------------------------------------------------

    def new_run_dir(self, vid: str) -> tuple:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{vid[:6]}"
        rid, n = base, 1
        while (self.root / "runs" / rid).exists():
            rid = f"{base}-{n}"
            n += 1
        path = self.root / "runs" / rid
        path.mkdir(parents=True)
        return rid, path

    def run_dir(self, rid: str) -> Path:
        if not re.fullmatch(r"[\w\-]+", rid):
            raise KeyError(rid)
        path = self.root / "runs" / rid
        if not path.is_dir():
            raise KeyError(rid)
        return path

    def run_meta(self, rid: str) -> Optional[Dict[str, Any]]:
        try:
            return _read_json(self.run_dir(rid) / "meta.json", None)
        except KeyError:
            return None

    def save_run_meta(self, rid: str, meta: Dict[str, Any]) -> None:
        _write_json(self.run_dir(rid) / "meta.json", meta)

    def runs(self) -> List[Dict[str, Any]]:
        out = []
        for d in sorted((self.root / "runs").iterdir(), reverse=True):
            if d.is_dir():
                meta = _read_json(d / "meta.json", None)
                if meta:
                    out.append(meta)
        return out

    def delete_run(self, rid: str) -> None:
        shutil.rmtree(self.run_dir(rid), ignore_errors=True)

    def last_runs_by_video(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for meta in self.runs():  # newest first
            vid = meta.get("video_id")
            if vid and vid not in out and meta.get("status") == "done":
                out[vid] = meta
        return out


def write_json(path: Path, payload: Any) -> None:
    _write_json(path, payload)


def read_json(path: Path, default: Any = None) -> Any:
    return _read_json(path, default)


__all__ = [
    "DEFAULT_SETTINGS", "VIDEO_SUFFIXES", "Workspace", "build_config", "clean_settings",
    "read_json", "safe_filename", "video_id", "write_json",
]
