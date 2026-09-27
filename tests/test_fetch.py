"""Videos from a link: looking one up, fetching a part, and the app's side.

No test here touches the network.  yt-dlp is replaced by a small script that
answers the way it does -- JSON for ``-J``, a file and ffmpeg-style progress
lines for a download -- so what is tested is everything around it: the
choices passed to it, reading its progress, stopping it, and where the file
ends up.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from billiards import fetch

FAKE_YTDLP = textwrap.dedent(r'''
    import json, os, shutil, sys, time
    args = sys.argv[1:]
    mode = os.environ.get("FAKE_MODE", "ok")
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(args) + "\n")
    if "-J" in args:
        if mode == "fail":
            print("ERROR: [youtube] abc: Video unavailable", file=sys.stderr)
            sys.exit(1)
        info = {"id": "abc123", "title": "Final | Kaçi vs Szewczyk | 10-Ball World Championship",
                "duration": 3683, "webpage_url": "https://www.youtube.com/watch?v=abc123",
                "extractor_key": "Youtube", "uploader": "Box Billiards", "width": 1280,
                "height": 720, "fps": 60, "vcodec": "avc1.640020"}
        if mode == "playlist":
            info = {"_type": "playlist", "id": "PL1", "title": "a list"}
        if mode == "live":
            info["is_live"] = True
        print(json.dumps(info, ensure_ascii=False))
        sys.exit(0)
    out = args[args.index("-o") + 1].replace("%(ext)s", "mp4")
    steps = 40 if mode == "slow" else 3
    for i in range(1, steps + 1):
        sys.stdout.write(f"frame={i * 100} fps=500 time=00:00:{2 * i:05.2f} bitrate=1000kbits/s\r")
        sys.stdout.flush()
        time.sleep(0.1 if mode == "slow" else 0.01)
    if mode == "fail":
        print("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        sys.exit(1)
    shutil.copyfile(os.environ["FAKE_CLIP"], out)
    print("[download] 100% of 2.00MiB in 00:00:01")
''')


@pytest.fixture
def fake_ytdlp(tmp_path, monkeypatch, clip):
    script = tmp_path / "fake_yt_dlp.py"
    script.write_text(FAKE_YTDLP, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(fetch, "_command", lambda: [sys.executable, str(script)])
    monkeypatch.setattr(fetch, "missing", lambda: None)
    monkeypatch.setattr(fetch, "_ffmpeg", lambda: "ffmpeg")  # passed on, never run by the script
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLIP", str(clip))
    monkeypatch.setenv("FAKE_MODE", "ok")

    def calls():
        return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return calls


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    from make_synthetic_clip import generate

    path = tmp_path_factory.mktemp("clip") / "break.mp4"
    generate(path, seed=0, fps=30.0, duration_s=2.5, width=854, height=480)
    return path


# --------------------------------------------------------------------------
# The small parts
# --------------------------------------------------------------------------


def test_times_are_read_as_people_write_them():
    assert fetch.parse_clock("75") == 75.0
    assert fetch.parse_clock("20:00") == 1200.0
    assert fetch.parse_clock("1:02:03.5") == pytest.approx(3723.5)
    assert fetch.parse_clock("") is None and fetch.parse_clock(None) is None
    for bad in ("1:2:3:4", "ten", "-5", "1::2"):
        with pytest.raises(ValueError):
            fetch.parse_clock(bad)
    assert fetch.clock_label(1215) == "20:15" and fetch.clock_label(3683) == "1:01:23"
    assert (fetch.about(20), fetch.about(838), fetch.about(10278), fetch.about(7200)) == ("20 s", "14 min", "2 h 50 min", "2 h")


def test_the_game_is_read_from_the_title():
    assert fetch.suggest_numbers("2026 WPA Men's 10-Ball World Championship") == "1-10"
    assert fetch.suggest_numbers("US Open 9 Ball final") == "1-9"
    assert fetch.suggest_numbers("8-ball league night") == "1-15"
    assert fetch.suggest_numbers("Snooker highlights") is None


def test_the_estimate_is_frames_over_tracking_speed():
    # The 28 Sep measurement: 4 min of a 60 fps broadcast took 407 s in the app.
    assert fetch.estimate_tracking_s(240, 60, 35.41) == pytest.approx(406.7, abs=1)
    # So the hour-long final it came from would take about 1 h 45 min.
    assert fetch.about(fetch.estimate_tracking_s(3683, 60, fetch.TYPICAL_PROCESSING_FPS)) == "1 h 45 min"
    # From the command line, drawing the diagram into -o, nearly three hours.
    assert fetch.about(fetch.estimate_tracking_s(3683, 60, fetch.TYPICAL_PROCESSING_FPS_DIAGRAM)) == "2 h 50 min"


def test_file_names_say_what_and_which_part():
    info = {"title": "Highlight Final | Eklent Kaçi VS Wojciech Szewczyk | 2026 WPA Men’s 10-Ball World Championship",
            "id": "d5TyZPetBkA"}
    stem = fetch.file_stem(info, 1200, 1500)
    assert stem.endswith("- d5TyZPetBkA (20.00-25.00)") and "Kaçi" in stem
    assert not any(c in stem for c in '|:/\\?*"<>')
    assert fetch.file_stem(info, None, None).endswith("- d5TyZPetBkA")
    assert fetch.file_stem({"title": "a/b"}, 0, None).endswith("(0.00-end)")


def test_a_part_is_fetched_from_the_streamed_copy_at_720p():
    part, whole = fetch.format_selector(True), fetch.format_selector(False)
    assert part.split("/")[0] == "bv*[height<=720][vcodec^=avc1][protocol^=m3u8]"
    assert whole.split("/")[0] == "bv*[height<=720][vcodec^=avc1]" and "m3u8" not in whole


def test_progress_is_read_from_ytdlp_and_ffmpeg_lines():
    assert fetch.read_progress("[download]  45.2% of 22.00MiB at 3.02MiB/s ETA 00:04", None) == pytest.approx(0.452)
    assert fetch.read_progress("frame= 987 fps=638 time=00:02:00.00 bitrate=1514kbits/s", 240) == pytest.approx(0.5)
    assert fetch.read_progress("frame= 987 fps=638 time=00:02:00.00", None) is None
    assert fetch.read_progress("[youtube] abc: Downloading webpage", 240) is None


def test_only_web_links_are_links():
    assert fetch.is_link("https://www.youtube.com/watch?v=d5TyZPetBkA")
    assert not fetch.is_link("C:\\Videos\\match.mp4") and not fetch.is_link("file:///etc/passwd")


# --------------------------------------------------------------------------
# Looking up and downloading, with yt-dlp played by a script
# --------------------------------------------------------------------------


def test_a_link_is_looked_up(fake_ytdlp):
    info = fetch.look_up("https://www.youtube.com/watch?v=abc123&t=1200")
    assert info["title"].startswith("Final | Kaçi") and info["duration_s"] == 3683
    assert (info["height"], info["fps"], info["numbers"]) == (720, 60.0, "1-10")
    args = fake_ytdlp()[0]
    assert "--no-playlist" in args and args[-1] == "https://www.youtube.com/watch?v=abc123&t=1200"
    assert args[args.index("--") - 1].startswith("bv*[height<=720]")  # the same choice as the download


@pytest.mark.parametrize("mode, words", [("playlist", "one video"), ("live", "live stream"), ("fail", "Video unavailable")])
def test_links_that_cannot_be_used_say_why(fake_ytdlp, monkeypatch, mode, words):
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(fetch.LinkError, match=words):
        fetch.look_up("https://www.youtube.com/watch?v=abc123")
    with pytest.raises(fetch.LinkError, match="http"):
        fetch.look_up("C:/Videos/match.mp4")


def test_only_the_part_asked_for_is_downloaded(fake_ytdlp, tmp_path):
    seen = []
    path = fetch.download("https://www.youtube.com/watch?v=abc123", tmp_path / "dl", 1200, 1206,
                          on_progress=lambda f, text: seen.append(f))
    assert path.parent == tmp_path / "dl" and path.suffix == ".mp4" and path.stat().st_size > 1000
    assert "(20.00-20.06)" in path.name
    assert list((tmp_path / "dl" / ".incoming").iterdir()) == []
    args = fake_ytdlp()[-1]
    assert args[args.index("--download-sections") + 1] == "*1200.00-1206.00"
    assert "--ffmpeg-location" in args
    # ffmpeg's carriage-return progress, as fractions of the 6 s part, ending at 1.
    assert seen[0] == 0.0 and 0.3 < seen[1] < 0.4 and seen[-1] == 1.0 and seen == sorted(seen)


def test_the_whole_video_needs_no_part(fake_ytdlp, tmp_path):
    fetch.download("https://www.youtube.com/watch?v=abc123", tmp_path / "dl", 0, 4000)
    args = fake_ytdlp()[-1]
    assert "--download-sections" not in args and "m3u8" not in args[args.index("-f") + 1]


def test_a_download_can_be_stopped_and_leaves_nothing_behind(fake_ytdlp, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_MODE", "slow")
    t0 = time.time()
    with pytest.raises(fetch.Cancelled):
        fetch.download("https://www.youtube.com/watch?v=abc123", tmp_path / "dl", 60, 140,
                       should_stop=lambda: time.time() - t0 > 1.0)
    assert time.time() - t0 < 4.0, "stopping took as long as the download"
    assert [p for p in (tmp_path / "dl").rglob("*") if p.is_file()] == []


def test_a_failed_download_says_why(fake_ytdlp, monkeypatch, tmp_path):
    info = fetch.look_up("https://www.youtube.com/watch?v=abc123")
    monkeypatch.setenv("FAKE_MODE", "fail")
    with pytest.raises(fetch.LinkError, match="403.*pip install -U yt-dlp"):
        fetch.download(info["url"], tmp_path / "dl", 60, 70, info=info)
    assert [p for p in (tmp_path / "dl").rglob("*") if p.is_file()] == []
    with pytest.raises(fetch.LinkError, match="only 1:01:23 long"):
        fetch.download(info["url"], tmp_path, 4000, None, info=info)


# --------------------------------------------------------------------------
# The app: paste a link, pick a part, it is downloaded and then tracked
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_app_downloads_a_part_and_tracks_it(fake_ytdlp, tmp_path):
    import http.client
    import threading

    from billiards.app.server import make_server
    from billiards.app.workspace import Workspace

    srv, app = make_server(Workspace(tmp_path / "ws"), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    port = srv.server_address[1]

    def call(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                     headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"})
        res = conn.getresponse()
        data = json.loads(res.read() or b"null")
        conn.close()
        return res.status, data

    try:
        status, info = call("POST", "/api/links/look-up", {"url": "https://www.youtube.com/watch?v=abc123"})
        assert status == 200 and info["long"] and info["numbers"] == "1-10"
        # Nothing tracked here yet, so the estimate uses the typical speed.
        sp = info["speed"]
        assert sp["slow"] == sp["fast"] == fetch.TYPICAL_PROCESSING_FPS and sp["from"] == "a typical broadcast"
        assert info["whole_estimate_s"] == [round(3683 * 60 / fetch.TYPICAL_PROCESSING_FPS)] * 2

        status, err = call("POST", "/api/links/download", {"url": info["url"], "start": "20:00", "end": "19:00"})
        assert status == 400 and "after the start" in err["error"]
        status, job = call("POST", "/api/links/download", {
            "url": info["url"], "start": "20:00", "end": "20:03", "settings": {"numbers": "1-10"},
        })
        assert status == 201 and job["start_s"] == 1200 and job["end_s"] == 1203

        deadline = time.time() + 180
        while time.time() < deadline:
            _, lib = call("GET", "/api/videos")
            d = next(x for x in lib["downloads"] if x["id"] == job["id"])
            if d["status"] not in ("queued", "downloading"):
                break
            time.sleep(0.2)
        assert d["status"] == "done" and d["run_id"], d
        video = next(v for v in lib["videos"] if v["id"] == d["video_id"])
        assert Path(video["folder"]) == tmp_path / "ws" / "downloads"
        assert app.ws.settings(video["id"])["numbers"] == "1-10"

        while time.time() < deadline:
            _, run = call("GET", f"/api/runs/{d['run_id']}")
            if run["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.3)
        assert run["status"] == "done", run
        # Removing a downloaded video deletes the file: the app put it there.
        call("POST", "/api/videos/forget", {"id": video["id"]})
        assert not Path(video["path"]).exists()
        status, _ = call("POST", f"/api/downloads/{job['id']}/dismiss")
        _, lib = call("GET", "/api/videos")
        assert not any(x["id"] == job["id"] for x in lib["downloads"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_estimate_is_a_range_that_one_fast_run_cannot_shrink(tmp_path):
    """28 Sep: one minute with many close-ups tracked at 42.9 fps; taken alone it
    would make every estimate look short.  Until there are three runs, a
    broadcast's pace stays in the range."""
    from billiards.app.downloads import processing_fps
    from billiards.app.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    assert processing_fps(ws)["from"] == "a typical broadcast"

    def done(fps, frames=3600, live=False):
        rid, _ = ws.new_run_dir("0123456789ab")
        ws.save_run_meta(rid, {"id": rid, "status": "done", "live": live,
                               "brief": {"processing_fps": fps, "frames": frames}})

    done(42.9)
    done(90.0, frames=100)   # too short to say anything
    done(90.0, live=True)    # live runs skip frames on purpose
    sp = processing_fps(ws)
    assert (sp["slow"], sp["fast"]) == (fetch.TYPICAL_PROCESSING_FPS, 42.9)
    assert sp["from"] == "your last run and a typical broadcast"
    done(30.0)
    done(25.0)
    sp = processing_fps(ws)
    assert (sp["slow"], sp["fast"], sp["fps"]) == (25.0, 42.9, 30.0) and sp["from"] == "your last 3 runs"


def test_without_ytdlp_the_app_says_how_to_get_it(monkeypatch, tmp_path):
    from billiards.app.downloads import DownloadManager
    from billiards.app.workspace import Workspace

    monkeypatch.setattr(fetch, "missing", lambda: "Links need yt-dlp, which is not installed: pip install yt-dlp")
    manager = DownloadManager(Workspace(tmp_path / "ws"), runs=None)
    with pytest.raises(fetch.LinkError, match="pip install yt-dlp"):
        manager.look_up("https://www.youtube.com/watch?v=abc123")
