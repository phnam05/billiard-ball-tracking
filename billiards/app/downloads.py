"""Links, for the app: look one up, fetch the part wanted, then track it.

A download is a job like a run: it waits its turn (one at a time, so two
downloads do not split the connection), reports how far it has got, and can be
stopped.  The file lands in the workspace's ``downloads/`` folder, which the
library lists, and unless told otherwise the new video is queued for tracking
the moment it is complete.

How long tracking will take is estimated before anything is fetched, from
this computer's own recent runs.  It is a range, not a number: the speed
depends on the footage as well as on the computer.  On 28 Sep 2026 four
minutes of a 60 fps broadcast tracked at 35.4 frames a second in the app, and
the first of those minutes alone at 42.9: a fifth of it was close-ups, where
the table is not in view and the tracker does next to nothing.
"""

from __future__ import annotations

import statistics
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .. import fetch
from .workspace import Workspace, video_id


def processing_fps(ws: Workspace, last: int = 5) -> Dict[str, Any]:
    """How fast this computer tracks, slowest to fastest, from its last runs.

    Until there are three runs to go on, the pace of a typical broadcast
    (``fetch.TYPICAL_PROCESSING_FPS``) is one of them, so a single lucky run
    cannot make the estimate for an hour of footage look short.
    """
    rates = []
    for meta in ws.runs():  # newest first
        brief = meta.get("brief") or {}
        fps = brief.get("processing_fps")
        if meta.get("status") == "done" and not meta.get("live") and fps and (brief.get("frames") or 0) >= 300:
            rates.append(float(fps))
        if len(rates) >= last:
            break
    if not rates:
        source = "a typical broadcast"
    elif len(rates) < 3:
        source = f"your last {'run' if len(rates) == 1 else f'{len(rates)} runs'} and a typical broadcast"
    else:
        source = f"your last {len(rates)} runs"
    if len(rates) < 3:
        rates.append(fetch.TYPICAL_PROCESSING_FPS)
    return {
        "fps": round(statistics.median(rates), 1),
        "slow": round(min(rates), 1),
        "fast": round(max(rates), 1),
        "from": source,
    }


class DownloadJob:
    def __init__(self, info: Dict[str, Any], start_s: Optional[float], end_s: Optional[float],
                 track: bool, settings: Optional[Dict[str, Any]]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.info = info
        self.start_s = start_s
        self.end_s = end_s
        self.track = track
        self.settings = settings or {}
        self.status = "queued"
        self.message = "waiting for the download before it"
        self.error: Optional[str] = None
        self.progress = 0.0
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.video_id: Optional[str] = None
        self.video_name: Optional[str] = None
        self.run_id: Optional[str] = None
        self.cancel_event = threading.Event()

    @property
    def part_s(self) -> Optional[float]:
        duration = self.info.get("duration_s")
        end = self.end_s if self.end_s is not None else duration
        return None if end is None else max(0.0, end - (self.start_s or 0.0))

    def to_dict(self) -> Dict[str, Any]:
        eta = None
        if self.status == "downloading" and self.started and self.progress > 0.02:
            spent = time.time() - self.started
            eta = spent * (1.0 - self.progress) / self.progress
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "progress": round(self.progress, 4),
            "eta_s": None if eta is None else round(eta, 1),
            "title": self.info.get("title"),
            "thumbnail": self.info.get("thumbnail"),
            "url": self.info.get("url"),
            "site": self.info.get("site"),
            "start_s": self.start_s,
            "end_s": self.end_s,
            "part_s": self.part_s,
            "fps": self.info.get("fps"),
            "track": self.track,
            "created": self.created,
            "finished": self.finished,
            "video_id": self.video_id,
            "video_name": self.video_name,
            "run_id": self.run_id,
        }


class DownloadManager:
    """Fetches links one at a time, in the background, then hands them on."""

    #: Finished downloads kept on the list, so the page can say what happened.
    KEEP = 20

    def __init__(self, ws: Workspace, runs: Any) -> None:
        self.ws = ws
        self.runs = runs
        self._jobs: Dict[str, DownloadJob] = {}
        self._queue: Deque[DownloadJob] = deque()
        self._wake = threading.Condition()
        threading.Thread(target=self._work, name="download-worker", daemon=True).start()

    # -- asking ------------------------------------------------------------

    def look_up(self, url: str) -> Dict[str, Any]:
        info = fetch.look_up(url)
        speed = processing_fps(self.ws)
        info["speed"] = speed
        duration = info.get("duration_s")
        info["long"] = bool(duration and duration > fetch.LONG_VIDEO_S)
        info["suggested_part_s"] = fetch.SUGGESTED_PART_S
        if duration:
            info["whole_estimate_s"] = [
                round(fetch.estimate_tracking_s(duration, info["fps"], speed[k])) for k in ("fast", "slow")
            ]
        return info

    def submit(self, url: str, start_s: Any = None, end_s: Any = None, track: bool = True,
               settings: Optional[Dict[str, Any]] = None) -> DownloadJob:
        start = fetch.parse_clock(start_s)
        end = fetch.parse_clock(end_s)
        if start is not None and end is not None and end <= start:
            raise ValueError("the end must be after the start")
        if settings:
            from .workspace import clean_settings

            clean_settings(settings)  # refused now, not after the download
        info = fetch.look_up(url)
        duration = info.get("duration_s")
        if duration and start is not None and start >= duration:
            raise ValueError(f"the video is only {fetch.clock_label(duration)} long")
        if duration and end is not None and end > duration:
            end = None
        job = DownloadJob(info, start or None, end, bool(track), settings)
        with self._wake:
            self._jobs[job.id] = job
            self._queue.append(job)
            self._prune()
            self._wake.notify()
        return job

    def cancel(self, jid: str) -> bool:
        job = self._jobs.get(jid)
        if job is None:
            return False
        job.cancel_event.set()
        if job.status == "queued":
            job.status, job.message, job.finished = "cancelled", "stopped before it started", time.time()
        return True

    def dismiss(self, jid: str) -> bool:
        job = self._jobs.get(jid)
        if job is None or job.status in ("queued", "downloading"):
            return False
        self._jobs.pop(jid, None)
        return True

    def list(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)]

    def active(self) -> List[DownloadJob]:
        return [j for j in self._jobs.values() if j.status in ("queued", "downloading")]

    def job(self, jid: str) -> Optional[DownloadJob]:
        return self._jobs.get(jid)

    # -- doing -------------------------------------------------------------

    def _prune(self) -> None:
        done = sorted((j for j in self._jobs.values() if j.status not in ("queued", "downloading")),
                      key=lambda j: j.created)
        for j in done[: max(0, len(done) - self.KEEP)]:
            self._jobs.pop(j.id, None)

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._queue:
                    self._wake.wait()
                job = self._queue.popleft()
            if job.cancel_event.is_set():
                continue
            self._run(job)

    def _run(self, job: DownloadJob) -> None:
        job.status, job.message, job.started = "downloading", "starting the download", time.time()

        def progress(fraction: Optional[float], text: str) -> None:
            if fraction is not None:
                job.progress = fraction
            job.message = text

        try:
            path = fetch.download(
                job.info.get("url") or "", self.ws.root / "downloads", job.start_s, job.end_s,
                info=job.info, on_progress=progress, should_stop=job.cancel_event.is_set,
            )
            video = self.ws.video(video_id(path))
            if video is None or video.get("error"):
                raise fetch.LinkError(f"{path.name} was downloaded, but it cannot be read as a video")
            job.video_id, job.video_name = video["id"], video["name"]
            if job.settings:
                self.ws.save_settings(video["id"], job.settings)
            job.status, job.progress = "done", 1.0
            job.message = "downloaded"
            if job.track:
                run = self.runs.submit(video["id"])
                job.run_id = run.id
                job.message = "downloaded; tracking it"
        except fetch.Cancelled:
            job.status, job.message = "cancelled", "stopped"
        except Exception as exc:  # reported to the page, not raised
            job.status = "failed"
            job.error = job.message = str(exc) or exc.__class__.__name__
            if not isinstance(exc, fetch.LinkError):
                traceback.print_exc()
        finally:
            job.finished = time.time()
