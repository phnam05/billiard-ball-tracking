"""The app: its workspace, its web API, and live tracking.

The API tests run a real server on a free port against a short synthetic
clip, and talk to it over HTTP the way the page does.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from billiards.app.workspace import Workspace, build_config, clean_settings, safe_filename, video_id


# --------------------------------------------------------------------------
# Settings and the workspace
# --------------------------------------------------------------------------


def test_settings_are_checked_and_defaults_filled():
    s = clean_settings({"preset": "pool-7ft", "numbers": "1-9", "corners": [[0, 0], [10, 0], [10, 5], [0, 5]]})
    assert s["preset"] == "pool-7ft" and s["ball_set"] == "auto" and s["numbers"] == "1-9"
    assert s["corners"] == [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
    for bad in ({"preset": "pool-11ft"}, {"ball_set": "neon"}, {"numbers": "0-20"},
                {"corners": [[0, 0]]}, {"start_s": 5, "end_s": 2}, {"max_width": 100}):
        with pytest.raises(ValueError):
            clean_settings(bad)
    assert "junk" not in clean_settings({"junk": 1})


def test_settings_become_a_tracker_config():
    cfg = build_config(clean_settings({"preset": "snooker-12ft", "ball_set": "tv", "max_width": 960}))
    assert cfg.table.length_in == pytest.approx(140.5)
    assert cfg.balls.ball_set == "tv" and cfg.max_frame_width == 960
    # The app draws its own diagram, so its videos are the picture alone.
    assert cfg.render.overhead_panel is False and cfg.render.draw_hud is False
    # The far cushion is searched by default since 28 Sep 2026 (the ball model
    # turns down what else is there), and can be turned off.
    assert cfg.detector.search_raised_bed is True
    assert build_config(clean_settings({"far_cushion": False})).detector.search_raised_bed is False


def test_upload_names_are_made_safe():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("Trận đấu: bi-a?.mp4").endswith(".mp4")
    assert "/" not in safe_filename("a/b\\c.mov") and "\\" not in safe_filename("a/b\\c.mov")


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory) -> Path:
    from make_synthetic_clip import generate

    folder = tmp_path_factory.mktemp("footage")
    generate(folder / "break.mp4", seed=0, fps=30.0, duration_s=2.5, width=854, height=480)
    return folder


def test_a_folder_of_footage_is_listed_with_its_details(clip_dir, tmp_path):
    ws = Workspace(tmp_path / "ws", folders=[clip_dir])
    videos = ws.videos()
    assert [v["name"] for v in videos] == ["break.mp4"]
    v = videos[0]
    assert v["id"] == video_id(clip_dir / "break.mp4")
    assert (v["width"], v["height"]) == (854, 480) and v["frame_count"] > 60
    assert ws.thumbnail(v["id"]).stat().st_size > 1000
    # Settings are kept per video, and survive a restart.
    ws.save_settings(v["id"], {"preset": "pool-8ft"})
    assert Workspace(tmp_path / "ws").settings(v["id"])["preset"] == "pool-8ft"
    # Forgetting a video hides it without touching the file.
    ws.forget(v["id"])
    assert ws.videos() == [] and (clip_dir / "break.mp4").exists()


def test_a_file_that_is_not_a_video_is_refused(tmp_path):
    ws = Workspace(tmp_path / "ws")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    with pytest.raises(ValueError):
        ws.add_file(notes)
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not really a video")
    with pytest.raises(ValueError):
        ws.add_file(fake)


# --------------------------------------------------------------------------
# The web API
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(clip_dir, tmp_path_factory):
    from billiards.app.server import make_server

    ws = Workspace(tmp_path_factory.mktemp("ws"), folders=[clip_dir])
    srv, app = make_server(ws, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    yield srv, app
    app.live.stop()
    srv.shutdown()
    srv.server_close()


def _call(server, method, path, body=None, headers=None):
    srv, _ = server
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    kind = res.getheader("Content-Type") or ""
    return res.status, (json.loads(raw) if kind.startswith("application/json") and raw else raw), res


def test_the_page_and_its_status_are_served(server):
    status, page, _ = _call(server, "GET", "/")
    assert status == 200 and b"Billiard Tracker" in page
    status, js, res = _call(server, "GET", "/app.js")
    assert status == 200 and res.getheader("Content-Type").startswith("text/javascript")
    status, s, _ = _call(server, "GET", "/api/status")
    assert status == 200 and s["version"] and "pool-9ft" in s["presets"]


def test_requests_from_other_sites_are_refused(server):
    """A web page in the same browser must not be able to drive the app."""
    status, _, _ = _call(server, "GET", "/api/videos", headers={"Host": "evil.example:80"})
    assert status == 403
    status, _, _ = _call(server, "POST", "/api/videos/add", {"path": "C:/"},
                         headers={"Origin": "http://evil.example"})
    assert status == 403


def test_bad_requests_get_a_reason_not_a_crash(server):
    status, err, _ = _call(server, "GET", "/api/runs/does-not-exist")
    assert status == 404 and "error" in err
    status, err, _ = _call(server, "POST", "/api/runs", {"video_id": "000000000000"})
    assert status == 404
    _, vids, _ = _call(server, "GET", "/api/videos")
    vid = vids["videos"][0]["id"]
    status, err, _ = _call(server, "PUT", f"/api/videos/{vid}/settings", {"preset": "pool-11ft"})
    assert status == 400 and "preset" in err["error"]


@pytest.mark.slow
def test_the_table_is_checked_on_a_chosen_frame(server):
    _, vids, _ = _call(server, "GET", "/api/videos")
    vid = vids["videos"][0]["id"]
    status, rep, _ = _call(server, "POST", f"/api/videos/{vid}/check", {"t": 1.0})
    assert status == 200 and rep["ok"], rep
    assert len(rep["corners"]) == 4 and rep["image_size"] == [854, 480]
    assert rep["detections"], "no balls seen on the check frame"
    # Corners placed by hand are used as given.
    corners = rep["corners"]
    status, rep2, _ = _call(server, "POST", f"/api/videos/{vid}/check",
                            {"t": 1.0, "settings": {"corners": corners}})
    assert rep2["ok"] and rep2["manual"]
    assert max(abs(a - b) for p, q in zip(rep2["corners"], corners) for a, b in zip(p, q)) < 1.0
    status, img, res = _call(server, "GET", f"/api/videos/{vid}/frame.jpg?t=1.0")
    assert status == 200 and res.getheader("Content-Type") == "image/jpeg" and img[:2] == b"\xff\xd8"


@pytest.mark.slow
def test_a_video_is_tracked_and_its_results_served(server):
    _, vids, _ = _call(server, "GET", "/api/videos")
    vid = vids["videos"][0]["id"]
    status, job, _ = _call(server, "POST", "/api/runs", {"video_id": vid})
    assert status == 201
    rid = job["id"]
    deadline = time.time() + 180
    while time.time() < deadline:
        _, state, _ = _call(server, "GET", f"/api/runs/{rid}")
        if state["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.3)
    assert state["status"] == "done", state
    assert state["summary"]["frames_processed"] > 60

    _, viewer, _ = _call(server, "GET", f"/api/runs/{rid}/viewer.json")
    assert viewer["table"]["length_in"] == 100.0 and len(viewer["table"]["pockets"]) == 6
    assert len(viewer["tracks"]) >= 5
    t = viewer["tracks"][0]
    assert len(t["f"]) == len(t["x"]) == len(t["y"]) == len(t["v"]) == len(t["o"])
    assert t["colour"].startswith("#")

    # The video can be seeked in, as a <video> element does.
    status, part, res = _call(server, "GET", f"/api/runs/{rid}/video", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(part) == 100 and res.getheader("Content-Range").startswith("bytes 0-99/")
    status, csv_bytes, _ = _call(server, "GET", f"/api/runs/{rid}/files/tracks.csv")
    assert status == 200 and csv_bytes.startswith(b"frame,")
    # Listed, with a one-line brief, and deletable.
    _, runs, _ = _call(server, "GET", "/api/runs")
    assert any(r["id"] == rid and r["brief"]["tracks"] >= 5 for r in runs["runs"])
    _, vids, _ = _call(server, "GET", "/api/videos")
    assert vids["videos"][0]["last_run"]["id"] == rid
    status, _, _ = _call(server, "DELETE", f"/api/runs/{rid}")
    assert status == 200
    status, _, _ = _call(server, "GET", f"/api/runs/{rid}")
    assert status == 404


@pytest.mark.slow
def test_a_run_can_be_stopped_and_keeps_what_it_tracked(server):
    _, vids, _ = _call(server, "GET", "/api/videos")
    vid = vids["videos"][0]["id"]
    _, job, _ = _call(server, "POST", "/api/runs", {"video_id": vid})
    rid = job["id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        _, state, _ = _call(server, "GET", f"/api/runs/{rid}")
        if state.get("frames_done", 0) >= 30 or state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    _call(server, "POST", f"/api/runs/{rid}/cancel")
    while time.time() < deadline:
        _, state, _ = _call(server, "GET", f"/api/runs/{rid}")
        if state["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert state["status"] in ("cancelled", "done"), state
    status, viewer, _ = _call(server, "GET", f"/api/runs/{rid}/viewer.json")
    assert status == 200 and viewer["tracks"]


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_video_replayed_live_is_tracked_and_saved(server, clip_dir, tmp_path_factory):
    """A file played at its own pace stands in for a camera: frames arrive on
    their clock, the table is found from the first ones, and the session is
    saved as a run when the file ends."""
    from make_synthetic_clip import generate

    _, app = server
    longer = tmp_path_factory.mktemp("live") / "longer.mp4"
    generate(longer, seed=1, fps=30.0, duration_s=4.5, width=854, height=480)
    vid = app.ws.add_file(longer)["id"]
    status, state, _ = _call(server, "POST", "/api/live/start", {
        "source": {"kind": "file", "video_id": vid, "speed": 1.0},
        "settings": {"max_width": 854}, "record": True,
    })
    assert status == 200
    seen_tracking = False
    deadline = time.time() + 90
    while time.time() < deadline:
        _, state, _ = _call(server, "GET", "/api/live")
        seen_tracking |= state["status"] == "tracking"
        if not state["running"]:
            break
        time.sleep(0.2)
    assert seen_tracking and state["status"] == "ended", state
    assert state["frames_processed"] > 20 and state["run_id"]
    _, meta, _ = _call(server, "GET", f"/api/runs/{state['run_id']}")
    assert meta["status"] == "done" and meta["live"] is True
    status, viewer, _ = _call(server, "GET", f"/api/runs/{state['run_id']}/viewer.json")
    assert status == 200 and viewer["tracks"]


def test_a_live_source_that_does_not_open_says_so(server):
    status, _, _ = _call(server, "POST", "/api/live/start", {
        "source": {"kind": "url", "url": "http://127.0.0.1:1/nothing-here.mjpg"}, "settings": {},
    })
    assert status == 200
    deadline = time.time() + 30
    while time.time() < deadline:
        _, state, _ = _call(server, "GET", "/api/live")
        if not state["running"]:
            break
        time.sleep(0.2)
    assert state["status"] == "error" and "stream" in state["message"]
