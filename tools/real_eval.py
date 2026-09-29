#!/usr/bin/env python
"""Score tracking on real footage against hand-marked answer keys.

``tools/evaluate.py`` scores the simulator's clips, where every ball's position
is known.  Real clips have no such truth, and until 28 Sep 2026 they were judged
by counting track ids -- which that day compared two different things (the
older code restarted its ids at every camera change, so its "25 balls" were
334 tracks).  The answer keys in ``tools/truth/`` (format:
``tools/truth/README.md``) list, on keyframes a few seconds apart, every ball
that is in the picture, where it is and which one it is, plus what each stretch
of the clip shows (play, replay, close-up, crowd, titles).  This tracks each
clip the way the app does and reports:

* **found** (recall) and **real** (precision) -- of the balls marked, how many
  had a track on them; of the tracks drawn, how many were on a ball;
* **named right / wrong** -- of the balls found whose number is known, how
  many carried that number (``CUE``, ``8``, ``3`` ...), and how many a
  *wrong* one, which is worse than none;
* **ids per ball** -- how many different track ids one ball had over a rack
  (1 is right; every extra one is the ball "forgotten" and found as new), and
  **swaps** -- ids that were on two different balls;
* **phantoms** per keyframe -- tracks on nothing: hands, chalk, pockets;
* **paused** -- the share of play the tracker drew nothing on, and
  **off-play** -- the share of replays, close-ups, crowd and title frames it
  drew balls on;
* one **score**, ``1 - (missed + phantoms + extra ids) / balls``, the same
  shape as MOTA.

Usage::

    python tools/real_eval.py                      # every answer key
    python tools/real_eval.py --only us-open-3min
    python tools/real_eval.py --code ../other-checkout --out tmp/old
    python tools/real_eval.py --reuse results/real  # re-score the last runs

Tracks and run JSONs go to ``results/real/`` (not committed); ``--render``
also writes the tracked videos there.  It needs the clips: each key's
``source.file``, else it is downloaded from ``source.url`` (yt-dlp), with a
warning, because a new download of a part need not start on the same frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRUTH_DIR = ROOT / "tools" / "truth"
OUT = ROOT / "results" / "real"

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    sys.path.insert(0, str(ROOT))
    from billiards.assignment import _hungarian as linear_sum_assignment  # type: ignore

Ball = Union[int, str]

#: Tracks the clip in a fresh interpreter, with ``billiards`` imported from
#: the checkout being scored, the way the app sets it up.
_RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from billiards import RunOptions, run
from billiards.app.workspace import build_config, clean_settings
cfg = build_config(clean_settings(json.loads(sys.argv[3])))
opts = dict(video=sys.argv[2], export_csv=sys.argv[4], export_json=sys.argv[5], progress_every=0)
if sys.argv[6]:
    from billiards.video import browser_codec
    writer, suffix = browser_codec()
    opts.update(output=sys.argv[6] + suffix, writer=writer)
run(cfg, RunOptions(**opts))
"""


# --------------------------------------------------------------------------
# Answer keys and clips
# --------------------------------------------------------------------------


def load_truth(path: Path) -> Dict[str, Any]:
    truth = json.loads(Path(path).read_text(encoding="utf-8"))
    truth.setdefault("name", Path(path).stem)
    return truth


def all_truths(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    keys = [load_truth(p) for p in sorted(TRUTH_DIR.glob("*.json"))]
    return [k for k in keys if not only or k["name"] in only]


def clip_path(truth: Dict[str, Any]) -> Path:
    """The clip the key was marked on, downloading it if it is not here."""
    src = truth["source"]
    path = ROOT / src["file"]
    if path.exists():
        return path
    sys.path.insert(0, str(ROOT))
    from billiards import fetch

    start, end = (None if t is None else fetch.parse_clock(t) for t in (src.get("part") or (None, None)))
    print(f"[real] {truth['name']}: {src['file']} missing, downloading {src['url']} -- "
          "check the frames still line up with the key", flush=True)
    got = fetch.download(src["url"], path.parent / "incoming", start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    got.replace(path)
    return path


def track(truth: Dict[str, Any], code_root: Path, out_dir: Path, render: bool = False) -> Tuple[Path, Path]:
    """Track the clip with the app's settings and the code at ``code_root``."""
    # Absolute: the run starts in ``code_root``.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = truth["name"]
    csv_path, json_path = out_dir / f"{name}_tracks.csv", out_dir / f"{name}_run.json"
    video = out_dir / f"{name}_tracked" if render else ""
    subprocess.run(
        [sys.executable, "-c", _RUNNER, str(code_root), str(clip_path(truth)),
         json.dumps(truth.get("settings", {})), str(csv_path), str(json_path), str(video)],
        check=True, cwd=str(code_root),
    )
    return csv_path, json_path


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def read_rows(csv_path: Path) -> Dict[int, List[dict]]:
    """Every reported ball, by frame."""
    rows: Dict[int, List[dict]] = defaultdict(list)
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows[int(row["frame"])].append(row)
    return rows


def identity(row: dict) -> Optional[Ball]:
    """Which ball a reported row says it is, or None if it names none."""
    label = (row.get("label") or "").strip()
    if label == "CUE":
        return "cue"
    if label == "8":
        return 8
    number = (row.get("number") or "").strip()
    if number:
        return int(float(number))
    if label.isdigit():
        return int(label)
    return None


def _segment_index(truth: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    last = max(s["to"] for s in truth["segments"])
    kinds = sorted({s["kind"] for s in truth["segments"]})
    kind_of = np.full(last + 1, -1, dtype=np.int16)
    rack_of = np.zeros(last + 1, dtype=np.int16)
    for s in truth["segments"]:
        kind_of[s["from"]:s["to"] + 1] = kinds.index(s["kind"])
        rack_of[s["from"]:s["to"] + 1] = int(s.get("rack") or 0)
    return kind_of, rack_of, kinds


def score(truth: Dict[str, Any], rows: Dict[int, List[dict]],
          summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compare a run's rows with an answer key.  See the module docstring."""
    gate = float(truth.get("match_px", 16))
    kind_of, rack_of, kinds = _segment_index(truth)

    balls = found = phantoms = 0
    right = wrong = known = 0
    ids_of: Dict[Tuple[int, Ball], set] = defaultdict(set)
    balls_of: Dict[Tuple[int, str], set] = defaultdict(set)
    per_frame: List[Dict[str, Any]] = []

    for kf in truth["keyframes"]:
        f = int(kf["frame"])
        rack = int(rack_of[f]) if f < len(rack_of) else 0
        marked = [(b[0], np.array([float(b[1]), float(b[2])])) for b in kf["balls"]]
        ignore = [np.array([float(p[0]), float(p[1])]) for p in kf.get("ignore", [])]
        here = rows.get(f, [])
        xy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in here]).reshape(-1, 2)
        pairs: List[Tuple[int, int]] = []
        if marked and len(here):
            dist = np.linalg.norm(np.array([p for _, p in marked])[:, None, :] - xy[None], axis=2)
            ri, ci = linear_sum_assignment(np.where(dist <= gate, dist, 1e6))
            pairs = [(int(i), int(j)) for i, j in zip(ri, ci) if dist[i, j] <= gate]
        used = {j for _, j in pairs}
        extra = [
            j for j in range(len(here)) if j not in used
            and not any(float(np.linalg.norm(xy[j] - p)) <= gate for p in ignore)
        ]
        balls += len(marked)
        found += len(pairs)
        phantoms += len(extra)
        for i, j in pairs:
            name, row = marked[i][0], here[j]
            if name == "?":
                continue
            ids_of[(rack, name)].add(row["track_id"])
            balls_of[(rack, row["track_id"])].add(name)
            known += 1
            said = identity(row)
            if said == name:
                right += 1
            elif said is not None:
                wrong += 1
        per_frame.append({"frame": f, "balls": len(marked), "found": len(pairs), "phantoms": len(extra)})

    extra_ids = sum(len(v) - 1 for v in ids_of.values())
    swaps = sum(1 for v in balls_of.values() if len(v) > 1)

    # What was drawn where nothing should be, and where it should have been.
    drawn = np.zeros(len(kind_of), dtype=bool)
    for f in rows:
        if 0 <= f < len(drawn):
            drawn[f] = True
    play = kinds.index("play") if "play" in kinds else -1
    off_play = {
        k: round(float(drawn[kind_of == i].mean()), 3)
        for i, k in enumerate(kinds) if k != "play" and np.any(kind_of == i)
    }
    paused = round(float(1.0 - drawn[kind_of == play].mean()), 3) if play >= 0 else None

    def frac(a: int, b: int) -> Optional[float]:
        return round(a / b, 3) if b else None

    out: Dict[str, Any] = {
        "keyframes": len(truth["keyframes"]),
        "balls": balls,
        "score": round(1.0 - (balls - found + phantoms + extra_ids) / balls, 3) if balls else None,
        "found": frac(found, balls),
        "real": frac(found, found + phantoms),
        "named_right": frac(right, known),
        "named_wrong": frac(wrong, known),
        "ids_per_ball": round(float(np.mean([len(v) for v in ids_of.values()])), 2) if ids_of else None,
        "swaps": swaps,
        "phantoms_per_keyframe": round(phantoms / max(1, len(truth["keyframes"])), 2),
        "paused": paused,
        "off_play": off_play,
        "track_ids": len({r["track_id"] for rs in rows.values() for r in rs}),
        "per_keyframe": per_frame,
    }
    if summary:
        out["tracks_created"] = summary.get("tracks_created")
        out["shots"] = len(summary.get("shot_log") or [])
        out["processing_fps"] = summary.get("processing_fps")
    if truth.get("shots") is not None:
        out["shots_marked"] = len(truth["shots"])
        if summary:
            fps = float((summary.get("video") or {}).get("fps") or 30.0)
            out.update(score_shots(truth["shots"], summary.get("shot_log") or [], fps))
    return out


#: A reported shot is a marked one if it starts within this many seconds of it.
SHOT_TOLERANCE_S = 1.5


def score_shots(marked: Sequence[Dict[str, Any]], reported: Sequence[Dict[str, Any]],
                fps: float) -> Dict[str, Any]:
    """How the shot log compares with the shots marked on the key.

    Reported and marked shots are paired by one assignment that keeps the
    starts as close as possible, and a pair counts if they are within
    ``SHOT_TOLERANCE_S``; matched in time order instead, a phantom shot
    during a dissolve 1.4 s before a real one took its place.
    **shots found** counts the marked shots matched, **extra** the reported
    shots that match none (a shot split in two, a ball nudged by hand, a
    replay), and **pots right** the matched shots whose balls potted, by
    number, are the ones marked (a scratch is left out: the keys do not
    mark the cue ball).  Two shots merged into one find only the first.
    """
    tol = SHOT_TOLERANCE_S * fps
    pairs: List[Tuple[int, int]] = []
    if marked and reported:
        gap = np.abs(np.array([[float(r["start_frame"]) - float(m["frame"]) for r in reported] for m in marked]))
        rows, cols = linear_sum_assignment(np.where(gap <= tol, gap, 1e9))
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if gap[i, j] <= tol]

    def numbers(potted: Sequence[Any]) -> set:
        return {int(p) for p in potted if str(p).isdigit()}

    right = sum(1 for i, j in pairs
                if numbers(reported[j].get("potted") or []) == numbers(marked[i].get("potted") or []))
    offsets = [(int(reported[j]["start_frame"]) - int(marked[i]["frame"])) / fps for i, j in pairs]
    return {
        "shots_found": len(pairs),
        "shots_extra": len(reported) - len(pairs),
        "pots_right": right,
        "shot_start_offset_s": round(float(np.median(offsets)), 2) if offsets else None,
    }


def measure(truth: Dict[str, Any], code_root: Path = ROOT, out_dir: Path = OUT,
            render: bool = False, reuse: bool = False) -> Dict[str, Any]:
    name = truth["name"]
    csv_path, json_path = out_dir / f"{name}_tracks.csv", out_dir / f"{name}_run.json"
    if not (reuse and csv_path.exists()):
        csv_path, json_path = track(truth, code_root, out_dir, render)
    summary = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
    return {"name": name, **score(truth, read_rows(csv_path), summary)}


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    def pct(v: Optional[float]) -> str:
        return "   -" if v is None else f"{100 * v:4.0f}"

    def shots(r: Dict[str, Any]) -> str:
        if r.get("shots_found") is None:
            return str(r.get("shots", "-"))
        return f"{r['shots_found']}/{r['shots_marked']} +{r['shots_extra']}"

    def pots(r: Dict[str, Any]) -> str:
        return "-" if r.get("pots_right") is None else str(r["pots_right"])

    print(f"\n{'clip':16s} {'score':>6s} {'found':>5s} {'real':>5s} {'named':>5s} {'wrong':>5s} "
          f"{'ids/ball':>8s} {'swaps':>5s} {'phant':>5s} {'paused':>6s} {'off-play':>8s} {'shots':>9s} "
          f"{'pots':>4s} {'fps':>5s}")
    for r in results:
        off = max(r["off_play"].values()) if r["off_play"] else 0.0
        print(f"{r['name']:16s} {r['score']:6.3f} {pct(r['found']):>5s} {pct(r['real']):>5s} "
              f"{pct(r['named_right']):>5s} {pct(r['named_wrong']):>5s} {r['ids_per_ball'] or 0:8.2f} "
              f"{r['swaps']:5d} {r['phantoms_per_keyframe']:5.2f} {pct(r['paused']):>6s} {pct(off):>8s} "
              f"{shots(r):>9s} {pots(r):>4s} {r.get('processing_fps') or 0:5.1f}")
    print("found/real/named/wrong/paused/off-play in %; off-play is the worst of replay/close-up/crowd/titles")
    print("shots: marked shots found / marked, + reported shots matching none; "
          f"pots: found shots whose pots are right (within {SHOT_TOLERANCE_S:g} s of a marked shot)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="answer keys to score, by name")
    ap.add_argument("--code", default=str(ROOT), help="checkout whose billiards/ to run (default: this one)")
    ap.add_argument("--out", default=str(OUT), help="where tracks and run JSONs go")
    ap.add_argument("--render", action="store_true", help="also write the tracked videos")
    ap.add_argument("--reuse", action="store_true", help="score the tracks already in --out, don't re-run")
    ap.add_argument("--json", help="write the results here too")
    args = ap.parse_args(argv)

    results = []
    for truth in all_truths(args.only):
        print(f"[real] {truth['name']} ...", flush=True)
        results.append(measure(truth, Path(args.code).resolve(), Path(args.out), args.render, args.reuse))
    print_table(results)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
