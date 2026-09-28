#!/usr/bin/env python
"""How the tracker holds up on footage unlike the sample clips.

The three sample clips are one venue, one camera angle, one cloth.  This
renders the same simulated break under the conditions other footage brings --
another cloth colour, another camera position, another resolution or frame
rate, a screen recording -- tracks each with default settings, and scores it
against the simulator's ground truth, so "does it work on my footage?" has a
measured answer for each kind of footage rather than a hope.

Writes ``reports/robustness.json`` (the numbers), ``results/robustness.png``
(one tracked frame of every variant, to look at) and prints a table::

    python tools/robustness.py              # every variant
    python tools/robustness.py --only blue corner
    python tools/robustness.py --duration 3 # quicker, noisier
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from billiards import Config, RunOptions, run  # noqa: E402
from evaluate import (  # noqa: E402
    apply_symmetry, best_symmetry, evaluate, evaluate_events, load_ground_truth, load_speeds,
    load_tracks, load_visibility, symmetric_tracks,
)
from make_synthetic_clip import generate  # noqa: E402

#: (name, what it stands for, generate() arguments).  The first is the
#: benchmark every other differs from in one respect.
VARIANTS: List[tuple] = [
    ("baseline", "the sample broadcasts: end camera, blue-grey cloth, 1280x720, 30 fps", {}),
    ("green", "green cloth", {"cloth": "green"}),
    ("blue", "tournament-blue cloth", {"cloth": "blue"}),
    ("red", "burgundy cloth", {"cloth": "red"}),
    ("tan", "camel cloth", {"cloth": "tan"}),
    ("grey", "nearly neutral grey cloth", {"cloth": "grey"}),
    ("blue-floor", "the sample clips' cloth on a royal-blue floor, as at the 2026 US Open",
     {"floor": "blue"}),
    ("green-red-floor", "green cloth on a red floor", {"cloth": "green", "floor": "red"}),
    ("side", "camera across from a long rail", {"camera": "side"}),
    ("overhead", "camera straight down from the ceiling", {"camera": "overhead", "cloth": "green"}),
    ("corner", "tripod at a corner of the room, table small in frame", {"camera": "corner", "cloth": "blue"}),
    ("480p", "854x480", {"width": 854, "height": 480}),
    ("1080p", "1920x1080", {"width": 1920, "height": 1080}),
    ("60fps", "filmed at 60 fps", {"fps": 60.0}),
    ("screen-rec", "25 fps broadcast screen-recorded at 37.5 fps, 1 frame in 8 missed",
     {"fps": 25.0, "container_fps": 37.5, "drop_rate": 0.12, "capture_jitter": 0.6}),
    ("tv-set", "the broadcasts' ball set (4 pink, 5 purple, black caps)", {"ball_set": "tv"}),
]


def _number_accuracy(gt_path: Path, tracks_path: Path, symmetry: str = "same",
                     gate_in: float = 2.25) -> Optional[float]:
    """Of the numbered balls matched to a track, how many carry the right number."""
    from scipy.optimize import linear_sum_assignment

    gt: Dict[int, list] = defaultdict(list)
    with gt_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if float(r.get("visible", 1) or 1) >= 0.5:
                gt[int(r["frame"])].append((int(r.get("number") or 0), float(r["x_in"]), float(r["y_in"])))
    tr: Dict[int, list] = defaultdict(list)
    with tracks_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n = r.get("number")
            label = r["label"]
            number = int(n) if n else (8 if label == "8" else 0 if label == "CUE" else -1)
            x, y = apply_symmetry(float(r["x_in"]), float(r["y_in"]), symmetry, 100.0, 50.0)
            tr[int(r["frame"])].append((number, x, y))
    right = total = 0
    for f, balls in gt.items():
        found = tr.get(f)
        if not found:
            continue
        a = np.array([[b[1], b[2]] for b in balls])
        b = np.array([[t[1], t[2]] for t in found])
        cost = np.linalg.norm(a[:, None] - b[None], axis=2)
        rows, cols = linear_sum_assignment(np.where(cost <= gate_in, cost, 1e6))
        for i, j in zip(rows, cols):
            if cost[i, j] > gate_in or balls[i][0] in (0, 8):
                continue  # the cue ball and the 8 are roles, scored elsewhere
            total += 1
            right += int(found[j][0] == balls[i][0])
    return round(right / total, 3) if total else None


def measure(name: str, what: str, kwargs: Dict[str, Any], work: Path, duration: float) -> Dict[str, Any]:
    video, gt, ev = work / f"{name}.mp4", work / f"{name}_gt.csv", work / f"{name}_events.json"
    tracks = work / f"{name}_tracks.csv"
    generate(video, seed=0, duration_s=duration, ground_truth_path=gt, events_path=ev, **kwargs)
    shot: Dict[str, Any] = {}
    frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    want = int(frames * 0.45)

    def grab(result, pipeline) -> None:
        if result.frame_index == want and result.annotated is not None:
            shot["frame"] = result.annotated.copy()

    t0 = time.time()
    try:
        summary = run(Config().apply_preset(), RunOptions(
            video=str(video), export_csv=str(tracks), progress_every=0, on_frame=grab, annotate=True,
        ))
    except (RuntimeError, ValueError) as exc:
        return {"name": name, "what": what, "calibrated": False, "error": str(exc)}
    truth, found = load_ground_truth(gt), load_tracks(tracks)
    # The simulated table is 100 x 50 in, as the default preset.
    symmetry = best_symmetry(truth, found)
    report = evaluate(
        truth, symmetric_tracks(found, symmetry), gate_in=2.25,
        gt_speeds=load_speeds(gt, "ball"), track_speeds=load_speeds(tracks, "track_id"),
        gt_visibility=load_visibility(gt),
    )
    run_events = []
    for e in summary["event_log"]:
        x, y = apply_symmetry(e["x_in"], e["y_in"], symmetry, 100.0, 50.0)
        run_events.append({**e, "x_in": x, "y_in": y})
    events = evaluate_events(json.loads(ev.read_text(encoding="utf-8")), run_events,
                             fps=float(summary["video"]["fps"]))
    speed = report.get("speed_error") or {}
    return {
        "name": name,
        "what": what,
        "calibrated": True,
        "recall": report["recall"],
        "precision": report["precision"],
        "mota": report["mota"],
        "id_switches": report["id_switches"],
        "position_error_in": None if report["matched"] == 0 else report["position_error_in"]["median"],
        "speed_error": speed.get("median_relative"),
        "numbers_right": _number_accuracy(gt, tracks, symmetry),
        "table_turned": symmetry,
        "cushions": f"{events['cushion']['matched']}/{events['cushion']['true']} (+{events['cushion']['false_positives']})",
        "collisions": f"{events['collision']['matched']}/{events['collision']['true']} (+{events['collision']['false_positives']})",
        "ball_px": round(float(summary["calibration"]["table"]["ball_radius_px_at_centre"]) * 2, 1),
        "source_fps": summary["clock"]["source_fps"],
        "tracking_fps": round(summary["frames_processed"] / max(time.time() - t0, 1e-6), 1),
        "_frame": shot.get("frame"),
    }


def contact_sheet(rows: List[Dict[str, Any]], path: Path, tile_w: int = 480) -> None:
    tiles = []
    for r in rows:
        img = r.get("_frame")
        if img is None:
            img = np.full((270, 480, 3), 30, np.uint8)
            cv2.putText(img, "no table found", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((34, tile_w, 3), 24, np.uint8)
        text = r["name"] + (f"   MOTA {r['mota']:.2f}   recall {r['recall'] or 0:.2f}" if r.get("calibrated") else "   no table found")
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for t in tiles]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 24))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def _fmt(value: Any, spec: str, width: int, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value * scale if scale != 1.0 else value:{spec}}{suffix}".rjust(width)


def print_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print(f"{'variant':12s} {'MOTA':>6s} {'recall':>6s} {'prec':>6s} {'IDsw':>4s} {'pos in':>6s} "
          f"{'speed':>6s} {'numbers':>7s} {'ball px':>7s}  cushions    contacts")
    for r in rows:
        if not r["calibrated"]:
            print(f"{r['name']:12s} no table found: {r['error'][:80]}")
            continue
        print(f"{r['name']:12s} {_fmt(r['mota'], '.3f', 6)} {_fmt(r['recall'], '.3f', 6)} "
              f"{_fmt(r['precision'], '.3f', 6)} {_fmt(r['id_switches'], 'd', 4)} "
              f"{_fmt(r['position_error_in'], '.2f', 6)} {_fmt(r['speed_error'], '.1f', 6, 100, '%')} "
              f"{_fmt(r['numbers_right'], '.0f', 7, 100, '%')} {_fmt(r['ball_px'], '.1f', 7)}  "
              f"{r['cushions']:11s} {r['collisions']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="just these variants")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds of play per clip")
    ap.add_argument("--no-save", action="store_true", help="print only; do not write the report")
    args = ap.parse_args(argv)

    chosen = [v for v in VARIANTS if not args.only or v[0] in args.only]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, what, kwargs in chosen:
            print(f"[robustness] {name}: {what} ...", flush=True)
            rows.append(measure(name, what, kwargs, Path(tmp), args.duration))

    if not args.no_save:
        sheet = ROOT / "results" / "robustness.png"
        contact_sheet(rows, sheet)
        out = ROOT / "reports" / "robustness.json"
        payload = {
            "measured": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "duration_s": args.duration,
            "variants": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        }
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print_table(rows)
    if not args.no_save:
        print()
        print(f"[robustness] wrote {out.relative_to(ROOT)} and {sheet.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
