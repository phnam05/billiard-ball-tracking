#!/usr/bin/env python
"""Score a tracking run against ground truth.

Turns "the tracking felt better" into numbers.  Ground-truth balls are matched
to tracks frame by frame with optimal assignment inside a physical gate, which
is the standard MOT protocol, and the report gives:

* **recall / precision** -- how much of the truth was found, and how much of
  what was reported was real;
* **ID switches** -- how often a ball changed track id, the metric that actually
  captures "the tracker lost it during the collision";
* **position error** -- median and 95th percentile, in inches;
* **MOTA** -- the single combined number, ``1 - (FN + FP + IDSW) / GT``;
* **speed error** -- how far each matched ball's reported speed is from its
  true one, which is where a wrong clock shows up (every position can be right
  while every speed is 30% off);
* **events**, when the simulator's event log and the run's JSON are given --
  how many of the collisions, cushion contacts and pots that physically
  happened were reported, and how many reported ones did not happen.

Usage::

    python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv
    python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv \
        --events-gt out/events.json --run-json out/run.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from billiards.assignment import _hungarian as linear_sum_assignment  # type: ignore


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_ground_truth(path: Path) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    out: Dict[int, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for row in _read_csv(path):
        out[int(row["frame"])].append(
            (row["ball"], np.array([float(row["x_in"]), float(row["y_in"])]))
        )
    return out


def load_visibility(path: Path) -> Optional[Dict[int, Dict[str, float]]]:
    """``frame -> {ball: fraction visible}``, or None for ground truth without it."""
    rows = _read_csv(path)
    if not rows or "visible" not in rows[0]:
        return None
    out: Dict[int, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[int(row["frame"])][row["ball"]] = float(row["visible"])
    return out


def load_speeds(path: Path, key: str) -> Dict[int, Dict[object, float]]:
    """``frame -> {ball or track id: speed in in/s}``, from either CSV."""
    out: Dict[int, Dict[object, float]] = defaultdict(dict)
    for row in _read_csv(path):
        ident = row[key] if key == "ball" else int(row[key])
        out[int(row["frame"])][ident] = float(row["speed_in_s"])
    return out


def load_tracks(
    path: Path, states: Sequence[str] = ("confirmed",)
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    out: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    allowed = set(states)
    for row in _read_csv(path):
        if allowed and row.get("state") not in allowed:
            continue
        out[int(row["frame"])].append(
            (int(row["track_id"]), np.array([float(row["x_in"]), float(row["y_in"])]))
        )
    return out


#: The four ways a rectangle maps onto itself.  A pool table looks the same
#: turned end for end or mirrored across its long axis, so which corner a
#: tracker calls (0, 0) depends on where the camera stands: from behind an end
#: rail, like the sample broadcasts, it agrees with the simulator's; from a
#: long rail, the ceiling or a corner it can be any of these.  Every one is an
#: equally right answer, and scoring against the wrong one reads a working
#: tracker as a broken one.
SYMMETRIES = ("same", "flip_x", "flip_y", "turned")


def apply_symmetry(x: float, y: float, name: str, length_in: float, width_in: float) -> Tuple[float, float]:
    fx = name in ("flip_x", "turned")
    fy = name in ("flip_y", "turned")
    return (length_in - x if fx else x, width_in - y if fy else y)


def best_symmetry(
    gt: Dict[int, List[Tuple[str, np.ndarray]]],
    tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    length_in: float = 100.0,
    width_in: float = 50.0,
    gate_in: float = 2.25,
) -> str:
    """Which symmetry of the table puts the most tracks on true balls."""
    frames = sorted(set(gt) & set(tracks))[::3]
    best, best_hits = SYMMETRIES[0], -1
    for name in SYMMETRIES:
        hits = 0
        for f in frames:
            g = np.array([p for _, p in gt[f]])
            t = np.array([apply_symmetry(p[0], p[1], name, length_in, width_in) for _, p in tracks[f]])
            if not len(g) or not len(t):
                continue
            d = np.linalg.norm(g[:, None, :] - t[None, :, :], axis=2)
            hits += int(np.sum(d.min(axis=0) <= gate_in))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def symmetric_tracks(
    tracks: Dict[int, List[Tuple[int, np.ndarray]]], name: str, length_in: float = 100.0, width_in: float = 50.0
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    return {
        f: [(tid, np.array(apply_symmetry(p[0], p[1], name, length_in, width_in))) for tid, p in rows]
        for f, rows in tracks.items()
    }


#: Below this true speed a ball's speed error says nothing about the clock:
#: a ball at rest reads ~0 however wrong the frame interval is.
_SPEED_SCORED_ABOVE_IN_S = 5.0


def evaluate(
    gt: Dict[int, List[Tuple[str, np.ndarray]]],
    tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    gate_in: float = 2.25,
    gt_speeds: Optional[Dict[int, Dict[object, float]]] = None,
    track_speeds: Optional[Dict[int, Dict[object, float]]] = None,
    gt_visibility: Optional[Dict[int, Dict[str, float]]] = None,
    min_visible: float = 0.5,
) -> dict:
    """Score tracks against ground truth.

    With ``gt_visibility``, a ball less than ``min_visible`` in view -- hidden
    behind nearer balls -- is *ignored*, as MOT benchmarks ignore occluded
    targets: missing it is not a miss, and a track on it is not a false
    positive.  No tracker can report what is not in the picture.
    """
    frames = sorted(gt)
    total_gt = 0
    total_pred = 0
    matches = 0
    errors: List[float] = []
    id_switches = 0
    last_id: Dict[str, int] = {}
    #: Every track id each ball was matched to.  One per ball is the ideal: a
    #: ball that becomes a new one at every camera cut has one per shot.
    ids_of: Dict[str, set] = defaultdict(set)
    per_ball_matched: Dict[str, int] = defaultdict(int)
    per_ball_total: Dict[str, int] = defaultdict(int)
    speed_abs: List[float] = []
    speed_rel: List[float] = []

    ignored_gt = 0
    for f in frames:
        g = gt.get(f, [])
        t = tracks.get(f, [])
        vis = gt_visibility.get(f, {}) if gt_visibility is not None else {}
        hidden = {name for name, _ in g if vis.get(name, 1.0) < min_visible}
        ignored_gt += len(hidden)
        total_gt += len(g) - len(hidden)
        total_pred += len(t)
        for name, _ in g:
            if name not in hidden:
                per_ball_total[name] += 1

        if not g or not t:
            continue

        gp = np.array([p for _, p in g])
        tp = np.array([p for _, p in t])
        cost = np.linalg.norm(gp[:, None, :] - tp[None, :, :], axis=2)
        cost_gated = np.where(cost <= gate_in, cost, 1e6)
        rows, cols = linear_sum_assignment(cost_gated)

        for r, c in zip(rows, cols):
            if cost_gated[r, c] >= 1e6:
                continue
            if g[r][0] in hidden:
                # A track on a hidden ball is neither credit nor blame.
                total_pred -= 1
                continue
            matches += 1
            errors.append(float(cost[r, c]))
            name = g[r][0]
            tid = t[c][0]
            per_ball_matched[name] += 1
            ids_of[name].add(tid)
            if name in last_id and last_id[name] != tid:
                id_switches += 1
            last_id[name] = tid

            if gt_speeds is not None and track_speeds is not None:
                true = gt_speeds.get(f, {}).get(name)
                est = track_speeds.get(f, {}).get(tid)
                if true is not None and est is not None and true > _SPEED_SCORED_ABOVE_IN_S:
                    speed_abs.append(abs(est - true))
                    speed_rel.append(abs(est - true) / true)

    fn = total_gt - matches
    fp = total_pred - matches
    mota = 1.0 - (fn + fp + id_switches) / total_gt if total_gt else float("nan")
    err = np.array(errors) if errors else np.array([np.nan])

    speed: Optional[dict] = None
    if speed_abs:
        a, r = np.array(speed_abs), np.array(speed_rel)
        speed = {
            "median_in_s": round(float(np.median(a)), 2),
            "p95_in_s": round(float(np.percentile(a, 95)), 2),
            "median_relative": round(float(np.median(r)), 4),
            "samples": len(speed_abs),
            "scored_above_in_s": _SPEED_SCORED_ABOVE_IN_S,
        }

    return {
        "frames": len(frames),
        "gt_instances": total_gt,
        "predicted_instances": total_pred,
        "matched": matches,
        "false_negatives": fn,
        "false_positives": fp,
        "id_switches": id_switches,
        "recall": round(matches / total_gt, 4) if total_gt else None,
        "precision": round(matches / total_pred, 4) if total_pred else None,
        "mota": round(float(mota), 4),
        "position_error_in": {
            "median": round(float(np.nanmedian(err)), 3),
            "mean": round(float(np.nanmean(err)), 3),
            "p95": round(float(np.nanpercentile(err, 95)), 3),
            "max": round(float(np.nanmax(err)), 3),
        },
        "ids_per_ball": round(float(np.mean([len(v) for v in ids_of.values()])), 2) if ids_of else None,
        "per_ball_recall": {
            name: round(per_ball_matched[name] / per_ball_total[name], 3)
            for name in sorted(per_ball_total)
        },
        "speed_error": speed,
        "gate_in": gate_in,
        "ignored_hidden_instances": ignored_gt,
        "min_visible": min_visible if gt_visibility is not None else None,
    }


#: A contact this gentle is physically real but carries nothing a viewer would
#: miss -- a ball nudging a cushion at walking pace -- so it is neither required
#: nor held against a detector that reports it.
_EVENT_MIN_SPEED_IN_S = {"collision": 15.0, "cushion": 15.0, "pot": 0.0}


def evaluate_events(
    gt_events: Sequence[dict],
    run_events: Sequence[dict],
    fps: float,
    time_tol_s: float = 0.2,
    dist_tol_in: float = 12.0,
) -> dict:
    """Match reported events to the ones the physics actually resolved.

    A reported event matches a true one of the same type if it is reported
    within ``time_tol_s`` of the first frame that shows it and within
    ``dist_tol_in`` of where it happened -- generous, because a cushion contact
    is reported where the ball was when the bounce became observable, which at
    break speed is several inches off the rail.  Each true event matches at
    most one reported one.  True events gentler than
    ``_EVENT_MIN_SPEED_IN_S`` are optional: matching one is not rewarded and
    reporting one is not penalised.
    """
    out: Dict[str, dict] = {}
    for kind, min_speed in _EVENT_MIN_SPEED_IN_S.items():
        speed_key = "closing_speed_in_s" if kind == "collision" else "speed_in_s"
        truth = [e for e in gt_events if e["type"] == kind and e.get("frame") is not None]
        preds = [e for e in run_events if e["type"] == kind]
        # Two balls already touching when the frame began -- momentum passing
        # through a rack on the break -- collide without anything a camera
        # could see: neither moves toward the other.  Optional, like a gentle
        # contact.
        required = [
            float(e.get(speed_key, min_speed)) >= min_speed and not e.get("touching_before")
            for e in truth
        ]

        used = set()
        matched_required = 0
        false_pos = 0
        for p in sorted(preds, key=lambda e: e["frame"]):
            best, best_cost = None, None
            for i, g in enumerate(truth):
                if i in used:
                    continue
                dt = abs(p["frame"] - g["frame"]) / fps
                dd = float(np.hypot(p["x_in"] - g["x_in"], p["y_in"] - g["y_in"]))
                if dt > time_tol_s or dd > dist_tol_in:
                    continue
                cost = dt / time_tol_s + dd / dist_tol_in
                if best_cost is None or cost < best_cost:
                    best, best_cost = i, cost
            if best is None:
                false_pos += 1
                continue
            used.add(best)
            if required[best]:
                matched_required += 1

        n_required = sum(required)
        out[kind] = {
            "true": n_required,
            "true_including_gentle": len(truth),
            "reported": len(preds),
            "matched": matched_required,
            "false_positives": false_pos,
            "recall": round(matched_required / n_required, 3) if n_required else None,
            "precision": round((len(preds) - false_pos) / len(preds), 3) if preds else None,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="ground-truth CSV")
    p.add_argument("--tracks", required=True, help="tracker CSV")
    p.add_argument("--gate", type=float, default=2.25,
                   help="match gate in inches (one ball diameter)")
    p.add_argument("--states", default="confirmed",
                   help="comma-separated track states to score, or 'all'")
    p.add_argument("--min-visible", type=float, default=0.5,
                   help="ignore ground-truth balls less than this fraction in view")
    p.add_argument("--events-gt", help="simulator event log (make_synthetic_clip.py --events)")
    p.add_argument("--run-json", help="the run's JSON (main.py track --json), for its event log")
    p.add_argument("--json", dest="out_json", help="write the report as JSON")
    args = p.parse_args()

    states = () if args.states == "all" else tuple(args.states.split(","))
    report = evaluate(
        load_ground_truth(Path(args.gt)),
        load_tracks(Path(args.tracks), states),
        gate_in=args.gate,
        gt_speeds=load_speeds(Path(args.gt), "ball"),
        track_speeds=load_speeds(Path(args.tracks), "track_id"),
        gt_visibility=load_visibility(Path(args.gt)),
        min_visible=args.min_visible,
    )
    if args.events_gt and args.run_json:
        run_json = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
        report["events"] = evaluate_events(
            json.loads(Path(args.events_gt).read_text(encoding="utf-8")),
            run_json["event_log"],
            fps=float(run_json["video"]["fps"]),
        )
    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
