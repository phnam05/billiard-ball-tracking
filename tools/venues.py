#!/usr/bin/env python
"""How the tracker does at other venues, on real footage.

``robustness.py`` answers "what if the cloth, camera or frame rate were
different?" on simulated footage, where every answer can be scored.  It could
not answer the question that mattered on 28 Sep 2026: the three sample clips
are one tournament, and a clip from the 2026 US Open -- a blue-grey cloth on a
royal-blue floor -- tracked nothing, because the floor was taken for the
cloth.  Real footage from other venues finds what a simulator does not think
of, so this tracks one minute from each of a list of venues and reports what
can be judged without ground truth:

* was the table found, which colour was taken for the cloth, and were the
  cushion noses found on all four rails;
* how much of the minute the table was in view;
* how many balls, shots, contacts, cushions and pots were reported.

The parts are downloaded once, into ``.cache/venues/`` (not committed), and
tracked the way the app tracks.  It writes ``reports/venues.json``, the
tracked minutes as ``results/venues/<name>.mp4``, and one frame of each, the
one with the most balls on it, as ``results/venues.png``::

    python tools/venues.py                  # every venue
    python tools/venues.py --only us-open mosconi
    python tools/venues.py --list

It needs yt-dlp, like the app's links.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from billiards import RunOptions, fetch, run  # noqa: E402
from billiards.app.workspace import build_config, clean_settings  # noqa: E402
from billiards.video import browser_codec  # noqa: E402

#: name, what is different about it, link (or a file in the repo), part
#: (start, end), table preset, balls in play.  Chosen on 28 Sep 2026 to differ
#: from the sample clips -- one WPA event, blue cloth, TV ball set -- in
#: venue, cloth, lighting, production and game.
VENUES: List[Dict[str, Any]] = [
    {"name": "sample", "what": "albin_fedor.mp4: the sample clips' venue, for comparison",
     "url": "albin_fedor.mp4", "part": (None, None), "preset": "pool-9ft", "numbers": "1-15"},
    {"name": "wpa-2026", "what": "2026 WPA 10-ball final, Box Billiards: the first link tried",
     "url": "https://www.youtube.com/watch?v=d5TyZPetBkA", "part": ("20:00", "21:00"),
     "preset": "pool-9ft", "numbers": "1-10"},
    {"name": "us-open", "what": "2026 US Open: blue-grey cloth on a royal-blue floor",
     "url": "https://www.youtube.com/watch?v=nuyq2hp9rLI", "part": ("11:48", "12:48"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "mosconi", "what": "2025 Mosconi Cup, day one: arena, team event",
     "url": "https://www.youtube.com/watch?v=aD2Q6PXwQGU", "part": ("5:00", "6:00"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "premier-league", "what": "2026 Premier League Pool final",
     "url": "https://www.youtube.com/watch?v=FOepzCM0aJ4", "part": ("5:00", "6:00"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "uk-open", "what": "2025 UK Open final",
     "url": "https://www.youtube.com/watch?v=vQzly86Nf2M", "part": ("10:00", "11:00"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "hanoi-open", "what": "2025 Hanoi Open: a Vietnamese production",
     "url": "https://www.youtube.com/watch?v=sKqEOgr6Byc", "part": ("5:00", "6:00"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "heyball", "what": "2024 JOY Heyball Masters: Chinese 8-ball, its own table and balls",
     "url": "https://www.youtube.com/watch?v=fI2_6S7Qjds", "part": ("20:00", "21:00"),
     "preset": "pool-9ft", "numbers": "1-15"},
    {"name": "derby-city", "what": "2016 Derby City Classic, Accu-Stats: older footage, another camera",
     "url": "https://www.youtube.com/watch?v=tzuOFj7JmIM", "part": ("20:00", "21:00"),
     "preset": "pool-9ft", "numbers": "1-9"},
    {"name": "bar-box", "what": "an amateur 8-ball tournament on a 7 ft bar table",
     "url": "https://www.youtube.com/watch?v=wIBiOC0PV-I", "part": ("5:00", "6:00"),
     "preset": "pool-7ft", "numbers": "1-15"},
    {"name": "snooker", "what": "2026 Wuhan Open snooker: 12 ft green table, snooker balls",
     "url": "https://www.youtube.com/watch?v=fV2UHRBui_4", "part": ("10:00", "11:00"),
     "preset": "snooker-12ft", "numbers": "1-15"},
]

CACHE = ROOT / ".cache" / "venues"
OUT = ROOT / "results" / "venues"


def footage(venue: Dict[str, Any]) -> Path:
    """The venue's minute, downloaded once."""
    url = venue["url"]
    if not fetch.is_link(url):
        return ROOT / url
    start, end = (fetch.parse_clock(t) for t in venue["part"])
    done = sorted(CACHE.glob(f"{venue['name']}.*"))
    if done:
        return done[0]
    got = fetch.download(url, CACHE / "incoming", start, end)
    dest = CACHE / f"{venue['name']}{got.suffix}"
    got.replace(dest)
    return dest


def measure(venue: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": venue["name"], "what": venue["what"], "url": venue["url"],
                           "part": venue["part"]}
    try:
        video = footage(venue)
    except (fetch.LinkError, OSError) as exc:
        return {**row, "calibrated": False, "error": f"download: {exc}"}
    cfg = build_config(clean_settings({"preset": venue["preset"], "numbers": venue["numbers"]}))
    writer, suffix = browser_codec()
    OUT.mkdir(parents=True, exist_ok=True)
    tracks_csv = OUT / f"{venue['name']}_tracks.csv"
    best: Dict[str, Any] = {"n": -1}

    def grab(result, pipeline) -> None:
        # The frame with the most balls on it, to show what was tracked.
        n = len(result.tracks)
        if result.annotated is not None and n > best["n"]:
            best.update(n=n, frame=result.annotated.copy(), index=result.frame_index)

    t0 = time.time()
    try:
        summary = run(cfg, RunOptions(
            video=str(video), output=str(OUT / f"{venue['name']}{suffix}"), writer=writer,
            export_csv=str(tracks_csv), progress_every=0, on_frame=grab, annotate=True,
        ))
    except (RuntimeError, ValueError) as exc:
        return {**row, "calibrated": False, "error": str(exc)}
    rows = Counter()
    with tracks_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["track_id"]] += 1
    calib = summary["calibration"]
    chosen = next((c for c in calib.get("cloth_candidates", []) if c.get("chosen")), {})
    noses = calib.get("cushion_noses") or {}
    frames = max(1, summary["frames_processed"])
    return {
        **row,
        "calibrated": True,
        "cloth": chosen.get("source"),
        "cloth_hsv": [calib["cloth"]["hue"], calib["cloth"]["sat"], calib["cloth"]["val"]],
        "cloth_score": chosen.get("score"),
        "rails_with_nose": sum(1 for n in noses.values() if n.get("of") and n["rays"] >= 0.4 * n["of"]),
        "ball_px": round(float(calib["table"]["ball_radius_px_at_centre"]) * 2, 1),
        "in_view": round(1.0 - summary["frames_view_lost"] / frames, 3),
        "recalibrations": summary["recalibrations"],
        "balls": sum(1 for n in rows.values() if n >= 15),
        "shots": len(summary.get("shot_log") or []),
        "events": summary["events"],
        "ball_set": summary.get("ball_set"),
        "tracking_fps": round(frames / max(time.time() - t0, 1e-6), 1),
        "_frame": best.get("frame"),
        "frame_shown": best.get("index"),
    }


def contact_sheet(rows: List[Dict[str, Any]], path: Path, tile_w: int = 480) -> None:
    tiles = []
    for r in rows:
        img = r.get("_frame")
        if img is None:
            img = np.full((270, tile_w, 3), 30, np.uint8)
            cv2.putText(img, "no table found", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((34, tile_w, 3), 24, np.uint8)
        text = r["name"] + (f"   {r['balls']} balls  {r['shots']} shots  in view {100 * r['in_view']:.0f}%"
                            if r.get("calibrated") else "   no table found")
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for t in tiles]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 24))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def print_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print(f"{'venue':15s} {'table':5s} {'noses':5s} {'view':>5s} {'balls':>5s} {'shots':>5s} "
          f"{'pots':>4s} {'cush':>4s} {'hits':>4s} {'ball px':>7s}  cloth")
    for r in rows:
        if not r["calibrated"]:
            print(f"{r['name']:15s} no    {r['error'][:90]}")
            continue
        e = r["events"]
        print(f"{r['name']:15s} {'yes':5s} {r['rails_with_nose']}/4   {100 * r['in_view']:4.0f}% {r['balls']:5d} "
              f"{r['shots']:5d} {e.get('pot', 0):4d} {e.get('cushion', 0):4d} {e.get('collision', 0):4d} "
              f"{r['ball_px']:7.1f}  {r['cloth']} (HSV {r['cloth_hsv'][0]:.0f}/{r['cloth_hsv'][1]:.0f}/{r['cloth_hsv'][2]:.0f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="*", help="venue names to run")
    p.add_argument("--list", action="store_true", help="list the venues and stop")
    args = p.parse_args()
    if args.list:
        for v in VENUES:
            print(f"{v['name']:15s} {v['what']}")
        return 0
    chosen = [v for v in VENUES if not args.only or v["name"] in args.only]
    rows = []
    for v in chosen:
        print(f"[venues] {v['name']}: {v['what']} ...", flush=True)
        rows.append(measure(v))
    print_table(rows)
    contact_sheet(rows, ROOT / "results" / "venues.png")
    report_path = ROOT / "reports" / "venues.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    kept = {r["name"]: r for r in previous.get("venues", [])}
    kept.update({r["name"]: {k: v for k, v in r.items() if not k.startswith("_")} for r in rows})
    report = {
        "what_this_is": "One minute from each of several venues, tracked with default settings. "
                        "No ground truth: see tools/venues.py for what each number means.",
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "venues": [kept[v["name"]] for v in VENUES if v["name"] in kept],
    }
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[venues] wrote {report_path.relative_to(ROOT)}, results/venues.png and results/venues/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
