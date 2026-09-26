"""The app's web server: a page in the browser, the tracker behind it.

Standard library only (``http.server``), so the app needs nothing the tracker
does not.  It listens on this computer only (127.0.0.1) unless told otherwise,
and refuses requests whose Host or Origin is not the app itself, so a web page
open in the same browser cannot drive it.

Routes (all JSON unless noted)::

    GET  /                               the page
    GET  /api/status                     version, workspace, video writer
    GET  /api/videos                     the library, with each video's last run
    POST /api/videos/add        {path}   a video file, or a folder of them
    POST /api/videos/forget     {id}
    PUT  /api/upload?name=      (bytes)  a video dropped on the page
    GET  /api/folders                    folders scanned for videos
    POST /api/folders/remove    {path}
    GET  /api/browse?path=               folders and videos, for the picker
    GET  /api/videos/<id>/thumb.jpg      (image)
    GET  /api/videos/<id>/frame.jpg?t=   (image) a frame at processing size
    GET  /api/videos/<id>/mask.jpg?t=    (image) what is taken for cloth
    GET  /api/videos/<id>/settings
    PUT  /api/videos/<id>/settings  {..}
    POST /api/videos/<id>/check     {settings, t}   calibration on one frame
    POST /api/runs              {video_id, settings?}
    GET  /api/runs                       every run, newest first
    GET  /api/runs/<id>                  state, live while running
    POST /api/runs/<id>/cancel
    DELETE /api/runs/<id>
    GET  /api/runs/<id>/preview.mjpg     (stream) the frames as they are made
    GET  /api/runs/<id>/viewer.json      every ball's path, for the results page
    GET  /api/runs/<id>/video            (video, seekable)
    GET  /api/runs/<id>/frame.jpg?f=     (image) for videos a browser cannot play
    GET  /api/runs/<id>/files/<name>     run.json, tracks.csv, the video
    GET  /api/live                       the live session's state
    GET  /api/live/cameras
    POST /api/live/start        {source, settings, record}
    POST /api/live/stop
    GET  /api/live/stream.mjpg           (stream)
    GET  /api/live/snapshot.jpg          (image) the camera's own picture
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from .. import __version__
from ..config import TABLE_PRESETS
from ..video import browser_codec
from .jobs import RunManager
from .live import LiveManager, list_cameras
from .preview import calibration_report, cloth_mask_jpeg
from .workspace import VIDEO_SUFFIXES, Workspace, clean_settings

STATIC = Path(__file__).resolve().parent / "static"


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class App:
    """Everything the handlers share."""

    def __init__(self, ws: Workspace, parallel: int = 1) -> None:
        self.ws = ws
        self.runs = RunManager(ws, parallel=parallel)
        self.live = LiveManager(ws)
        self.allowed_hosts: set = set()
        self._cameras: Optional[Tuple[float, List[Dict[str, Any]]]] = None

    def cameras(self, refresh: bool = False) -> List[Dict[str, Any]]:
        if refresh or self._cameras is None or time.time() - self._cameras[0] > 60:
            self._cameras = (time.time(), list_cameras())
        return self._cameras[1]


Route = Tuple[str, "re.Pattern[str]", Callable[..., Any]]


class Handler(BaseHTTPRequestHandler):
    server_version = f"billiards/{__version__}"
    app: App  # set on the class by ``make_server``
    routes: List[Route] = []

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("BILLIARDS_APP_LOG"):
            sys.stderr.write("[app] " + fmt % args + "\n")

    def _guard(self) -> None:
        host = (self.headers.get("Host") or "").lower()
        if self.app.allowed_hosts and host not in self.app.allowed_hosts:
            raise ApiError(HTTPStatus.FORBIDDEN, f"unexpected Host {host!r}")
        origin = self.headers.get("Origin")
        if origin:
            netloc = urlparse(origin).netloc.lower()
            if netloc not in self.app.allowed_hosts and netloc != host:
                raise ApiError(HTTPStatus.FORBIDDEN, "cross-origin request refused")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        self.query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            self._guard()
            for m, pattern, fn in self.routes:
                if m != method:
                    continue
                match = pattern.fullmatch(url.path)
                if match:
                    fn(self, *[unquote(g) for g in match.groups()])
                    return
            if method == "GET" and not url.path.startswith("/api/"):
                self._static(url.path)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such page: {url.path}")
        except ApiError as exc:
            self._json({"error": exc.message}, exc.status)
        except KeyError as exc:
            self._json({"error": f"not found: {exc.args[0] if exc.args else ''}"}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:  # pragma: no cover - reported, not raised
            traceback.print_exc()
            try:
                self._json({"error": f"{exc.__class__.__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:
                pass

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 4 << 20:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "the request body is not JSON")
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "the request body must be a JSON object")
        return data

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, content_type: Optional[str] = None, download: Optional[str] = None) -> None:
        """A file, honouring Range so a <video> can seek in it."""
        if not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such file: {path.name}")
        size = path.stat().st_size
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _mjpeg(self, preview: Any, still_running: Callable[[], bool]) -> None:
        """Frames as they come, as one multipart/x-mixed-replace response."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = -1
        while True:
            new_seq, jpeg = preview.wait(seq, timeout=5.0)
            if jpeg is not None and new_seq != seq:
                seq = new_seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
            if preview.closed and new_seq == seq:
                break
            if not still_running() and preview.closed:
                break

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC not in target.parents and target != STATIC / "index.html":
            raise ApiError(HTTPStatus.NOT_FOUND, "no such page")
        if not target.is_file():
            target = STATIC / "index.html"
        data = target.read_bytes()
        ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
                 ".svg": "image/svg+xml"}.get(target.suffix, "application/octet-stream")
        self._bytes(data, ctype + "; charset=utf-8" if ctype.startswith("text") else ctype, "no-cache")

    def num(self, key: str, default: Optional[float] = None) -> Optional[float]:
        value = self.query.get(key)
        if value in (None, ""):
            return default
        try:
            return float(value)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} must be a number")


def route(method: str, pattern: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        Handler.routes.append((method, re.compile(pattern), fn))
        return fn
    return deco


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _jpeg(img: Any, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the picture")
    return buf.tobytes()


# -- status and library -------------------------------------------------------


@route("GET", r"/api/status")
def status(h: Handler) -> None:
    writer, suffix = browser_codec()
    h._json({
        "version": __version__,
        "workspace": str(h.app.ws.root),
        "video_writer": writer,
        "browser_playable": writer != "mp4v",
        "presets": sorted(TABLE_PRESETS),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "active_runs": len(h.app.runs.active()),
        "live": h.app.live.state().get("status"),
    })


@route("GET", r"/api/videos")
def videos(h: Handler) -> None:
    last = h.app.ws.last_runs_by_video()
    active = {j.video["id"]: j.to_dict() for j in h.app.runs.active()}
    out = []
    for v in h.app.ws.videos():
        v = dict(v)
        v["last_run"] = last.get(v["id"])
        v["active_run"] = active.get(v["id"])
        v["has_settings"] = (h.app.ws.root / "settings" / f"{v['id']}.json").exists()
        out.append(v)
    h._json({"videos": out, "folders": h.app.ws.folders()})


@route("POST", r"/api/videos/add")
def add_video(h: Handler) -> None:
    raw = str(h.body().get("path") or "").strip().strip('"')
    if not raw:
        raise ValueError("give a file or folder path")
    path = Path(os.path.expandvars(raw)).expanduser()
    if path.is_dir():
        folder = h.app.ws.add_folder(path)
        h._json({"added_folder": folder, "videos": [v for v in h.app.ws.videos() if v["folder"] == folder]})
    else:
        h._json({"added": h.app.ws.add_file(path)})


@route("POST", r"/api/videos/forget")
def forget_video(h: Handler) -> None:
    h.app.ws.forget(str(h.body().get("id")))
    h._json({"ok": True})


@route("PUT", r"/api/upload")
def upload(h: Handler) -> None:
    name = h.query.get("name") or "upload.mp4"
    length = int(h.headers.get("Content-Length") or 0)
    if length <= 0:
        raise ValueError("empty upload")
    h._json({"added": h.app.ws.save_upload(name, h.rfile, length)})


@route("GET", r"/api/folders")
def folders(h: Handler) -> None:
    h._json({"folders": h.app.ws.folders()})


@route("POST", r"/api/folders/remove")
def remove_folder(h: Handler) -> None:
    h.app.ws.remove_folder(str(h.body().get("path")))
    h._json({"folders": h.app.ws.folders()})


@route("GET", r"/api/browse")
def browse(h: Handler) -> None:
    raw = h.query.get("path") or ""
    if not raw:
        roots = []
        if os.name == "nt":
            import string

            roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        else:
            roots = ["/"]
        home = str(Path.home())
        h._json({"path": "", "parent": None, "dirs": [home] + roots, "videos": [], "roots": True})
        return
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a folder: {path}")
    dirs, vids = [], []
    try:
        for p in sorted(path.iterdir(), key=lambda q: q.name.lower()):
            if p.name.startswith(".") or p.name.startswith("$"):
                continue
            try:
                if p.is_dir():
                    dirs.append(str(p))
                elif p.suffix.lower() in VIDEO_SUFFIXES:
                    vids.append({"path": str(p), "name": p.name, "size_bytes": p.stat().st_size})
            except OSError:
                continue
    except PermissionError:
        raise ValueError(f"no permission to read {path}")
    parent = str(path.parent) if path.parent != path else ""
    h._json({"path": str(path), "parent": parent, "dirs": dirs, "videos": vids, "roots": False})


@route("GET", r"/api/videos/([0-9a-f]{12})/thumb\.jpg")
def thumb(h: Handler, vid: str) -> None:
    h._file(h.app.ws.thumbnail(vid), "image/jpeg")


@route("GET", r"/api/videos/([0-9a-f]{12})/frame\.jpg")
def frame(h: Handler, vid: str) -> None:
    settings = h.app.ws.settings(vid)
    width = int(h.num("w", settings["max_width"]) or 0)
    img = h.app.ws.read_frame(vid, t_s=h.num("t", 0.0), max_width=width)
    h._bytes(_jpeg(img), "image/jpeg")


@route("GET", r"/api/videos/([0-9a-f]{12})/mask\.jpg")
def mask(h: Handler, vid: str) -> None:
    settings = h.app.ws.settings(vid)
    h._bytes(cloth_mask_jpeg(h.app.ws, vid, settings, h.num("t", 0.0) or 0.0), "image/jpeg")


@route("GET", r"/api/videos/([0-9a-f]{12})/settings")
def get_settings(h: Handler, vid: str) -> None:
    video = h.app.ws.video(vid)
    if video is None:
        raise KeyError(vid)
    h._json({"video": video, "settings": h.app.ws.settings(vid)})


@route("PUT", r"/api/videos/([0-9a-f]{12})/settings")
def put_settings(h: Handler, vid: str) -> None:
    if h.app.ws.video(vid) is None:
        raise KeyError(vid)
    h._json({"settings": h.app.ws.save_settings(vid, h.body())})


@route("POST", r"/api/videos/([0-9a-f]{12})/check")
def check(h: Handler, vid: str) -> None:
    body = h.body()
    settings = clean_settings({**h.app.ws.settings(vid), **(body.get("settings") or {})})
    t = body.get("t")
    h._json(calibration_report(h.app.ws, vid, settings, None if t is None else float(t)))


# -- runs ---------------------------------------------------------------------


@route("POST", r"/api/runs")
def start_run(h: Handler) -> None:
    body = h.body()
    job = h.app.runs.submit(str(body.get("video_id")), body.get("settings"))
    h._json(job.to_dict(), HTTPStatus.CREATED)


@route("GET", r"/api/runs")
def list_runs(h: Handler) -> None:
    h._json({"runs": h.app.runs.list()})


@route("GET", r"/api/runs/([\w\-]+)")
def get_run(h: Handler, rid: str) -> None:
    state = h.app.runs.describe(rid)
    if state is None:
        raise KeyError(rid)
    folder = h.app.ws.run_dir(rid)
    run_json = folder / "run.json"
    if state.get("status") in ("done", "cancelled") and run_json.exists():
        summary = json.loads(run_json.read_text(encoding="utf-8"))
        state["summary"] = {k: summary.get(k) for k in (
            "video", "calibration", "frames_processed", "wall_seconds", "processing_fps", "events",
            "recalibrations", "frames_view_lost", "frames_repeated", "clock", "shots",
            "tracks_created", "tracks_revived", "finished_reasons", "live", "frames_skipped",
        )}
    error = folder / "error.txt"
    if state.get("status") == "failed" and error.exists():
        state["traceback"] = error.read_text(encoding="utf-8")[-4000:]
    h._json(state)


@route("POST", r"/api/runs/([\w\-]+)/cancel")
def cancel_run(h: Handler, rid: str) -> None:
    h._json({"ok": h.app.runs.cancel(rid)})


@route("DELETE", r"/api/runs/([\w\-]+)")
def delete_run(h: Handler, rid: str) -> None:
    job = h.app.runs.job(rid)
    if job is not None and job.status in ("queued", "calibrating", "running", "finishing"):
        raise ValueError("this run is still going: cancel it first")
    h.app.ws.delete_run(rid)
    h._json({"ok": True})


@route("GET", r"/api/runs/([\w\-]+)/preview\.mjpg")
def run_preview(h: Handler, rid: str) -> None:
    job = h.app.runs.job(rid)
    if job is None:
        raise KeyError(rid)
    h._mjpeg(job.preview, lambda: job.status in ("queued", "calibrating", "running", "finishing"))


@route("GET", r"/api/runs/([\w\-]+)/viewer\.json")
def run_viewer(h: Handler, rid: str) -> None:
    h._file(h.app.ws.run_dir(rid) / "viewer.json", "application/json")


def _run_video(h: Handler, rid: str) -> Path:
    meta = h.app.runs.describe(rid) or {}
    name = meta.get("video_file")
    if not name:
        raise KeyError("video")
    return h.app.ws.run_dir(rid) / name


@route("GET", r"/api/runs/([\w\-]+)/video")
def run_video(h: Handler, rid: str) -> None:
    path = _run_video(h, rid)
    h._file(path, "video/webm" if path.suffix == ".webm" else "video/mp4")


@route("GET", r"/api/runs/([\w\-]+)/frame\.jpg")
def run_frame(h: Handler, rid: str) -> None:
    path = _run_video(h, rid)
    cap = cv2.VideoCapture(str(path))
    try:
        index = int(h.num("f", 0) or 0)
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise KeyError(f"frame {index}")
    width = int(h.num("w", 1280) or 0)
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    h._bytes(_jpeg(img, 80), "image/jpeg")


@route("GET", r"/api/runs/([\w\-]+)/files/([\w\-.]+)")
def run_file(h: Handler, rid: str, name: str) -> None:
    allowed = {"run.json", "tracks.csv", "viewer.json", "meta.json", "thumb.jpg", "error.txt",
               "tracked.mp4", "tracked.webm"}
    if name not in allowed:
        raise KeyError(name)
    h._file(h.app.ws.run_dir(rid) / name, download=None if name.endswith(".jpg") else f"{rid}-{name}")


# -- live ---------------------------------------------------------------------


@route("GET", r"/api/live")
def live_state(h: Handler) -> None:
    h._json(h.app.live.state())


@route("GET", r"/api/live/cameras")
def live_cameras(h: Handler) -> None:
    h._json({"cameras": h.app.cameras(refresh=h.query.get("refresh") == "1")})


@route("POST", r"/api/live/start")
def live_start(h: Handler) -> None:
    body = h.body()
    source = body.get("source") or {}
    if source.get("kind") not in ("camera", "url", "file"):
        raise ValueError("choose a camera, a stream address or a video")
    settings = clean_settings(body.get("settings") or {})
    session = h.app.live.start(source, settings, bool(body.get("record")))
    h._json(session.to_dict())


@route("POST", r"/api/live/stop")
def live_stop(h: Handler) -> None:
    h.app.live.stop()
    h._json(h.app.live.state())


@route("GET", r"/api/live/stream\.mjpg")
def live_stream(h: Handler) -> None:
    session = h.app.live.session
    if session is None:
        raise KeyError("no live session")
    h._mjpeg(session.preview, lambda: session.running)


@route("GET", r"/api/live/snapshot\.jpg")
def live_snapshot(h: Handler) -> None:
    session = h.app.live.session
    if session is None or not session.snapshot:
        raise KeyError("no picture yet")
    h._bytes(session.snapshot, "image/jpeg")


# -- serving ------------------------------------------------------------------


def make_server(ws: Workspace, host: str = "127.0.0.1", port: int = 8765,
                parallel: int = 1) -> Tuple[ThreadingHTTPServer, App]:
    app = App(ws, parallel=parallel)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    real_port = server.server_address[1]
    names = {host, "localhost", "127.0.0.1"} if host in ("127.0.0.1", "localhost") else {host}
    app.allowed_hosts = {f"{n}:{real_port}" for n in names}
    if host in ("0.0.0.0", "::"):
        app.allowed_hosts = set()  # listening everywhere: the user asked for it
    return server, app


def serve(workspace: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          folders: Tuple[Path, ...] = (), parallel: int = 1) -> int:
    ws = Workspace(workspace, folders=folders)
    try:
        server, app = make_server(ws, host, port, parallel)
    except OSError as exc:
        print(f"[billiards] could not listen on {host}:{port}: {exc}", file=sys.stderr)
        print("[billiards] another copy may be running; try --port 0 for any free port", file=sys.stderr)
        return 1
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{server.server_address[1]}/"
    print(f"[billiards] app running at {url}", flush=True)
    print(f"[billiards] workspace: {ws.root}", flush=True)
    writer, _ = browser_codec()
    if writer == "mp4v":
        print("[billiards] note: no video writer a browser can play was found; results are shown "
              "frame by frame. `pip install imageio-ffmpeg` fixes that.", flush=True)
    print("[billiards] press Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n[billiards] stopping", flush=True)
    finally:
        app.live.stop()
        server.server_close()
    return 0
