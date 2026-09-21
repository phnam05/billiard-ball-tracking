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
* **MOTA** -- the single combined number, ``1 - (FN + FP + IDSW) / GT``.

Usage::

    python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv
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


def evaluate(
    gt: Dict[int, List[Tuple[str, np.ndarray]]],
    tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    gate_in: float = 2.25,
) -> dict:
    frames = sorted(gt)
    total_gt = 0
    total_pred = 0
    matches = 0
    errors: List[float] = []
    id_switches = 0
    last_id: Dict[str, int] = {}
    per_ball_matched: Dict[str, int] = defaultdict(int)
    per_ball_total: Dict[str, int] = defaultdict(int)

    for f in frames:
        g = gt.get(f, [])
        t = tracks.get(f, [])
        total_gt += len(g)
        total_pred += len(t)
        for name, _ in g:
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
            matches += 1
            errors.append(float(cost[r, c]))
            name = g[r][0]
            tid = t[c][0]
            per_ball_matched[name] += 1
            if name in last_id and last_id[name] != tid:
                id_switches += 1
            last_id[name] = tid

    fn = total_gt - matches
    fp = total_pred - matches
    mota = 1.0 - (fn + fp + id_switches) / total_gt if total_gt else float("nan")
    err = np.array(errors) if errors else np.array([np.nan])

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
        "per_ball_recall": {
            name: round(per_ball_matched[name] / per_ball_total[name], 3)
            for name in sorted(per_ball_total)
        },
        "gate_in": gate_in,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="ground-truth CSV")
    p.add_argument("--tracks", required=True, help="tracker CSV")
    p.add_argument("--gate", type=float, default=2.25,
                   help="match gate in inches (one ball diameter)")
    p.add_argument("--states", default="confirmed",
                   help="comma-separated track states to score, or 'all'")
    p.add_argument("--json", dest="out_json", help="write the report as JSON")
    args = p.parse_args()

    states = () if args.states == "all" else tuple(args.states.split(","))
    report = evaluate(
        load_ground_truth(Path(args.gt)),
        load_tracks(Path(args.tracks), states),
        gate_in=args.gate,
    )
    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
