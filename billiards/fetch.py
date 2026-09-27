"""Videos from a link: YouTube, or any other site ``yt-dlp`` can read.

A match on YouTube is often an hour long, and tracking takes longer than the
video plays (a 60 fps broadcast tracks at a little over half its playing speed
in the app on a laptop, so an hour takes well over an hour and a half), so a
link is looked up first -- its title, length and frame rate, which is enough
to say how long tracking would take -- and then only the part wanted is
downloaded.

``yt-dlp`` is run as a separate process (``python -m yt_dlp``), not imported:
it can then be stopped mid-download together with the ``ffmpeg`` it starts,
and a site change that breaks it breaks one download, not the app.  It is an
optional dependency, like ``imageio-ffmpeg``.  Downloading *part* of a video
needs ``ffmpeg``; the one ``imageio-ffmpeg`` ships is used if there is none on
the PATH.

Choices made here, and why:

* **720p, H.264, no sound.**  The tracker works at 1280 px wide
  (``max_frame_width``), so a bigger picture only costs download and decoding
  time; OpenCV decodes H.264 everywhere, which is not true of AV1; the
  tracker never listens.
* **A part is fetched from the streamed (HLS) copy.**  On 28 Sep 2026
  YouTube refused (403) ``ffmpeg`` reading part of its direct file, while the
  streamed copy of the same 720p60 video gave a 4-minute part in 17 s.
* **Node.js is offered to yt-dlp** when Deno is missing: YouTube now needs a
  JavaScript runtime to list all its formats, and yt-dlp only looks for Deno
  unless told otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Tracking speed to estimate with when this computer has none of its own, in
#: frames a second.  Measured on 28 Sep 2026 on the work laptop, on 4 minutes
#: (20:00-24:00) of a 720p, 60 fps WPA broadcast with 32 camera changes: 14,401
#: frames in 407 s through the app, which writes its video with ffmpeg and
#: draws nothing below the picture.
TYPICAL_PROCESSING_FPS = 35.4
#: The same 4 minutes from the command line with ``-o``, which draws the
#: top-down diagram below every frame (674 s), and with ``--no-overhead`` too
#: (494 s).  Without ``-o`` nothing is drawn and it is at least as fast as the app.
TYPICAL_PROCESSING_FPS_DIAGRAM = 21.5
TYPICAL_PROCESSING_FPS_VIDEO = 29.2

#: Longer than this and the page suggests tracking a part.
LONG_VIDEO_S = 10 * 60
#: The part it suggests.
SUGGESTED_PART_S = 5 * 60

_HEIGHT = 720
#: Best to worst: H.264 up to 720p, anything but AV1 up to 720p, anything.
_FORMATS = (
    f"bv*[height<={_HEIGHT}][vcodec^=avc1]",
    f"bv*[height<={_HEIGHT}][vcodec!^=av01]",
    f"b[height<={_HEIGHT}]",
    "bv*",
    "b",
)


def format_selector(part: bool) -> str:
    """yt-dlp's ``-f``: for a part, the streamed copy of each choice first."""
    choices: List[str] = []
    for f in _FORMATS:
        if part and f.startswith("bv*["):
            choices.append(f + "[protocol^=m3u8]")
        choices.append(f)
    return "/".join(choices)


class LinkError(ValueError):
    """A link that cannot be used, with a reason a person can act on."""


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------


def is_link(text: str) -> bool:
    return bool(re.match(r"^https?://\S+$", (text or "").strip(), re.IGNORECASE))


def parse_clock(text: Any) -> Optional[float]:
    """Seconds from ``"75"``, ``"1:15"``, ``"1:02:03.5"``; None for blank."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        if text < 0:
            raise ValueError("a time cannot be negative")
        return float(text)
    s = str(text).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) > 3 or not all(re.fullmatch(r"\d+(\.\d*)?", p) for p in parts):
        raise ValueError(f"not a time: {text!r} (use seconds, m:ss or h:mm:ss)")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def clock_label(seconds: float, sep: str = ":") -> str:
    """``1215`` -> ``20:15``; ``3683`` -> ``1:01:23``."""
    s = int(round(seconds))
    h, m, sec = s // 3600, (s // 60) % 60, s % 60
    return f"{h}{sep}{m:02d}{sep}{sec:02d}" if h else f"{m}{sep}{sec:02d}"


def about(seconds: float) -> str:
    """A duration the way a person says it: ``20 s``, ``14 min``, ``2 h 50 min``."""
    if seconds < 90:
        return f"{int(round(seconds))} s"
    if seconds < 3600:
        return f"{int(round(seconds / 60))} min"
    m = 5 * int(round(seconds / 300))
    return f"{m // 60} h {m % 60} min" if m % 60 else f"{m // 60} h"


def suggest_numbers(title: str) -> Optional[str]:
    """The balls in play, when the title names the game (``10-Ball`` ...)."""
    m = re.search(r"\b(8|9|10)[\s-]?ball\b", title or "", re.IGNORECASE)
    return {"8": "1-15", "9": "1-9", "10": "1-10"}.get(m.group(1)) if m else None


def estimate_tracking_s(duration_s: float, fps: float, processing_fps: float) -> float:
    """How long tracking ``duration_s`` of video at ``fps`` would take."""
    return max(0.0, duration_s) * max(fps, 1.0) / max(processing_fps, 1e-6)


def file_stem(info: Dict[str, Any], start_s: Optional[float], end_s: Optional[float]) -> str:
    """A file name (no suffix) from the title, the video's id and the part."""
    # An apostrophe belongs to its word ("Men's"), anything else odd is a gap.
    title = re.sub(r"['’]", "", str(info.get("title") or "video"))
    title = re.sub(r"[^\w\-. ]+", " ", title, flags=re.UNICODE)
    title = re.sub(r"\s+", " ", title).strip(" .-_")
    if len(title) > 70:
        cut = title[:70]
        title = cut[: cut.rfind(" ")] if " " in cut[40:] else cut
    title = title.rstrip(" .-_") or "video"
    vid = re.sub(r"[^\w\-]+", "", str(info.get("id") or ""))[:20]
    stem = f"{title} - {vid}" if vid else title
    if start_s is not None or end_s is not None:
        a = clock_label(start_s or 0.0, ".")
        b = clock_label(end_s, ".") if end_s is not None else "end"
        stem += f" ({a}-{b})"
    return stem


# --------------------------------------------------------------------------
# yt-dlp
# --------------------------------------------------------------------------


def missing() -> Optional[str]:
    """Why links cannot be used on this computer, or None if they can."""
    if importlib.util.find_spec("yt_dlp") is None:
        return "Links need yt-dlp, which is not installed: pip install yt-dlp"
    return None


def _ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    from .video import _ffmpeg_exe

    return _ffmpeg_exe()


def _command() -> List[str]:
    """How yt-dlp is started (replaced in the tests)."""
    return [sys.executable, "-m", "yt_dlp"]


def _common_args() -> List[str]:
    args = ["--no-playlist", "--no-warnings", "--no-color"]
    if not shutil.which("deno"):
        for runtime in ("node", "bun"):
            if shutil.which(runtime):
                args += ["--js-runtimes", runtime]
                break
    return args


def _env() -> Dict[str, str]:
    # A title in Vietnamese or Polish must survive the trip through a pipe.
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _reason(output: str) -> str:
    """yt-dlp's own explanation, shortened, from what it printed."""
    errors = [l.strip() for l in output.splitlines() if l.strip().startswith("ERROR:")]
    text = errors[-1][len("ERROR:"):].strip() if errors else (output.strip().splitlines() or ["no output"])[-1]
    text = re.sub(r"^\[\w+\]\s*[\w-]+:\s*", "", text)
    return text[:300]


def look_up(url: str, timeout_s: float = 90.0) -> Dict[str, Any]:
    """What is at ``url``: title, length, and the picture that would be fetched."""
    why = missing()
    if why:
        raise LinkError(why)
    url = (url or "").strip()
    if not is_link(url):
        raise LinkError("paste a link that starts with http:// or https://")
    cmd = _command() + _common_args() + ["-J", "-f", format_selector(part=True), "--", url]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=timeout_s, env=_env())
    except subprocess.TimeoutExpired:
        raise LinkError("the site did not answer in time; try again")
    out = done.stdout.decode("utf-8", "replace")
    if done.returncode != 0 or not out.strip():
        raise LinkError(f"could not read the link: {_reason(done.stderr.decode('utf-8', 'replace') or out)}")
    try:
        raw = json.loads(out)
    except ValueError:
        raise LinkError("the site's answer could not be read")
    if raw.get("_type") in ("playlist", "multi_video"):
        raise LinkError("that is a playlist or a channel: paste the link of one video")
    if raw.get("is_live") or raw.get("live_status") in ("is_live", "is_upcoming"):
        raise LinkError("that is a live stream: open it in Live, or wait for the recording")
    return summarise(raw)


_SITE_NAMES = {"Youtube": "YouTube"}


def summarise(raw: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the app and the command line use, from yt-dlp's ``-J``."""
    title = str(raw.get("title") or raw.get("id") or "video")
    fmt = raw
    if raw.get("requested_formats"):
        videos = [f for f in raw["requested_formats"] if f.get("vcodec") not in (None, "none")]
        fmt = videos[0] if videos else raw
    duration = raw.get("duration")
    return {
        "url": raw.get("webpage_url") or raw.get("original_url"),
        "id": raw.get("id"),
        "site": _SITE_NAMES.get(raw.get("extractor_key"), raw.get("extractor_key") or raw.get("extractor")),
        "title": title,
        "uploader": raw.get("uploader") or raw.get("channel"),
        "duration_s": float(duration) if duration else None,
        "thumbnail": raw.get("thumbnail"),
        "width": fmt.get("width"),
        "height": fmt.get("height"),
        "fps": float(fmt.get("fps") or 30.0),
        "vcodec": fmt.get("vcodec"),
        "numbers": suggest_numbers(title),
    }


_PERCENT = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
_FFMPEG_TIME = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def read_progress(line: str, part_s: Optional[float]) -> Optional[float]:
    """The fraction done that one line of yt-dlp's (or ffmpeg's) output says."""
    m = _PERCENT.search(line)
    if m:
        return min(1.0, float(m.group(1)) / 100.0)
    m = _FFMPEG_TIME.search(line)
    if m and part_s:
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        return min(1.0, t / part_s)
    return None


def _kill(proc: subprocess.Popen) -> None:
    """Stop yt-dlp and the ffmpeg it started (killing Python alone leaves it)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


class Cancelled(Exception):
    pass


def download(
    url: str,
    folder: Path,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    info: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[Optional[float], str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """Fetch ``url`` (or its part from ``start_s`` to ``end_s``) into ``folder``.

    The file is written in ``folder/.incoming`` and moved into ``folder`` only
    when complete, so a library that lists ``folder`` never sees half a video.
    Returns the finished file.  Raises ``Cancelled`` if ``should_stop`` said so.
    """
    info = info or look_up(url)
    duration = info.get("duration_s")
    start_s = None if not start_s else float(start_s)
    end_s = None if end_s is None else float(end_s)
    if duration and end_s is not None and end_s >= duration - 0.5:
        end_s = None
    if duration and start_s is not None and start_s >= duration:
        raise LinkError(f"the video is only {clock_label(duration)} long")
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise LinkError("the end must be after the start")
    part = start_s is not None or end_s is not None
    if part and _ffmpeg() is None:
        raise LinkError("downloading part of a video needs ffmpeg: pip install imageio-ffmpeg")
    part_s = None
    if part:
        part_s = (end_s if end_s is not None else (duration or 0.0)) - (start_s or 0.0)

    folder = Path(folder)
    incoming = folder / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    stem = file_stem(info, start_s, end_s)
    for old in incoming.glob(f"{glob_escape(stem)}.*"):
        old.unlink(missing_ok=True)

    cmd = _command() + _common_args() + [
        "--newline", "--no-mtime", "--no-cache-dir",
        "-f", format_selector(part),
        "-o", str(incoming / (stem.replace("%", "%%") + ".%(ext)s")),
    ]
    ffmpeg = _ffmpeg()
    if ffmpeg:
        cmd += ["--ffmpeg-location", ffmpeg]
    if part:
        cmd += ["--download-sections", f"*{start_s or 0.0:.2f}-{'inf' if end_s is None else f'{end_s:.2f}'}"]
    cmd += ["--", info.get("url") or url]

    report = on_progress or (lambda fraction, text: None)
    report(0.0, "starting the download")
    popen_kw: Dict[str, Any] = {}
    if os.name != "nt":
        popen_kw["start_new_session"] = True
    else:
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_env(), **popen_kw)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def reader() -> None:
        # ffmpeg ends its progress lines with a carriage return, not a newline.
        buf = b""
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            *done, buf = re.split(rb"[\r\n]", buf)
            for raw in done:
                if raw.strip():
                    lines.put(raw.decode("utf-8", "replace"))
        if buf.strip():
            lines.put(buf.decode("utf-8", "replace"))
        lines.put(None)

    threading.Thread(target=reader, name="yt-dlp-output", daemon=True).start()
    tail: List[str] = []
    fraction = 0.0
    try:
        while True:
            if should_stop is not None and should_stop():
                _kill(proc)
                raise Cancelled()
            try:
                line = lines.get(timeout=0.3)
            except queue.Empty:
                continue
            if line is None:
                break
            if "Opening '" in line or "keepalive" in line or "reuse HTTP" in line:
                continue  # ffmpeg reporting each HLS segment
            tail = (tail + [line])[-30:]
            got = read_progress(line, part_s)
            if got is not None and got >= fraction:
                fraction = got
                report(fraction, "downloading")
        proc.wait()
    except BaseException:
        _kill(proc)
        for f in incoming.glob(f"{glob_escape(stem)}.*"):
            f.unlink(missing_ok=True)
        raise
    finished = [f for f in incoming.glob(f"{glob_escape(stem)}.*")
                if not f.name.endswith((".part", ".ytdl", ".temp")) and ".part" not in f.suffixes]
    if proc.returncode != 0 or not finished:
        for f in incoming.glob(f"{glob_escape(stem)}.*"):
            f.unlink(missing_ok=True)
        reason = _reason("\n".join(tail))
        hint = " If links used to work, yt-dlp may need updating: pip install -U yt-dlp" \
            if "Unsupported URL" not in reason else ""
        raise LinkError(f"the download failed: {reason}.{hint}".replace("..", "."))
    got = max(finished, key=lambda f: f.stat().st_size)
    dest = folder / got.name
    n = 1
    while dest.exists():
        dest = folder / f"{got.stem} ({n}){got.suffix}"
        n += 1
    os.replace(got, dest)
    report(1.0, "downloaded")
    return dest


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


__all__ = [
    "Cancelled", "LONG_VIDEO_S", "LinkError", "SUGGESTED_PART_S", "TYPICAL_PROCESSING_FPS",
    "TYPICAL_PROCESSING_FPS_DIAGRAM", "TYPICAL_PROCESSING_FPS_VIDEO", "about", "clock_label", "download", "estimate_tracking_s", "file_stem", "format_selector", "is_link",
    "look_up", "missing", "parse_clock", "read_progress", "suggest_numbers", "summarise",
]
