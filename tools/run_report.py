#!/usr/bin/env python
"""Track the sample clips, measure how noisy the output is, and log it.

``tools/evaluate.py`` answers "is the tracking correct?" against a synthetic
clip with exact ground truth.  This answers the question the real clips raise
instead, which is "how much of this output is junk?" -- phantom tracks, speed
estimates that swing by 50 in/s between frames, positions that jitter while
nothing is moving, events for things that did not happen.  There is no ground
truth for those clips, so the numbers here are not accuracy; they are
*self-consistency*, and physics says what self-consistent looks like: a rolling
ball decelerates smoothly, a ball at rest stays put, and a table with ten balls
on it does not produce sixty tracks.

Every run is appended to ``reports/run-log.json`` with the commit it came from
and a note saying what changed, so the next session can see where things stood
without re-deriving it.

It also **rewrites everything in ``results/``** -- the annotated video, the
per-frame CSV, the run JSON and the calibration preview for every clip -- so
that what is on disk to look at is always what the code currently does.  A
number in a log answers "did it get better?"; only the video answers "does it
look right?", and a stale video in ``results/`` reads as "nothing changed".

Usage::

    python tools/run_report.py --note "what I changed"
    python tools/run_report.py --note "..." --ground-truth   # also score MOTA
    python tools/run_report.py --note "..." --real           # also the real answer keys
    python tools/run_report.py --note "..." --no-render      # numbers only, faster
    python tools/run_report.py --show                        # print the log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from billiards.config import Config  # noqa: E402
from billiards.pipeline import RunOptions, run  # noqa: E402

LOG_PATH = ROOT / "reports" / "run-log.json"

#: The clips in the repo and what is actually in them.  This is the only ground
#: truth they have, and it was read off the broadcast's own rack graphic --
#: which ball numbers it still shows, and which one disappears -- rather than
#: guessed from the tracker's output -- and then checked against the frames
#: themselves, because the graphic is operated by hand and lags the table by
#: seconds.  ``tracks_reported`` above ``balls_visible`` means phantoms; a
#: ``pot`` count away from ``detectable_pots`` means phantom or missed pots.
#: ``real_pots`` can be higher: a ball already sitting in the pocket jaws is
#: inside the detector's pocket-exclusion zone, so it is never tracked and its
#: pot cannot be seen.
SAMPLE_CLIPS: Dict[str, Dict[str, Any]] = {
    "fedor_shot": {
        "video": "fedor_shot.mp4",
        "balls_visible": 8,
        "real_pots": 1,
        "detectable_pots": 1,
        "notes": "Rack graphic shows 2,3,4,6,7,8,9 -- seven objects plus the "
                 "cue. One shot: the cue ball runs the length of the table, "
                 "hits the blue 2-ball into the near-right corner, then comes "
                 "off several rails and settles. The clip cuts to another "
                 "angle at the very end.",
    },
    "albin_fedor": {
        "video": "albin_fedor.mp4",
        "balls_visible": 7,
        "real_pots": 2,
        "detectable_pots": 2,
        "notes": "Two pots, checked frame by frame. (1) The pink 4-ball sits "
                 "in the near-left pocket jaws from the start and the cue ball, "
                 "jumped over the 6, knocks it in at about frame 155; it is "
                 "inside the pocket-exclusion zone, and until 29 Sep 2026 was "
                 "never tracked (now found there with the ball model, "
                 "detect._balls_in_pockets). (2) The purple 5-ball is struck at about frame "
                 "555 and drops into the far-left corner at frame 566-568; "
                 "this one is fully visible and should be reported. The rack "
                 "graphic lags both: it drops the 4 somewhere between frames "
                 "200 and 400, and only starts fading the 5 at frame 750. "
                 "(Until 23 Sep 2026 this entry said the clip had one pot, "
                 "untrackable, so any reported pot was a phantom -- wrong: the "
                 "graphic had been read without checking the table.)",
    },
    "fedor_jump": {
        "video": "fedor_jump.mp4",
        "balls_visible": 9,
        "real_pots": 0,
        "detectable_pots": 0,
        "notes": "Rack graphic shows 2,3,4,5,6,7,8,9 throughout -- eight "
                 "objects plus the cue, and nothing is potted.",
    },
}

METRICS_MEANING: Dict[str, str] = {
    "tracks_reported": "ids that actually reached the video and the CSV; compare with balls_visible, anything above it is a phantom",
    "tracks_created": "ids handed out, including tentative ones that were dropped before they were ever drawn; always >= tracks_reported",
    "short_tracks": "tracks that lived under 15 frames -- these are the phantoms, want []",
    "frames_repeated": "input frames that were copies of the one before them and so were replayed, not measured",
    "speed_jitter_in_s": "mean/max change in a ball's estimated speed between consecutive frames while it is moving; a rolling ball decelerates smoothly, so lower is better",
    "peak_speed_in_s": "fastest speed any ball was credited with; a hard pool break is ~200 in/s and a normal shot 60-120, so a spike well above the ball's average is noise",
    "rest_jitter_in": "mean distance a ball at rest is reported to move per frame; want well under a ball radius (1.125 in)",
    "events": "event counts for the whole clip -- read against shots, which says what a player would have seen",
    "processing_fps": "speed, for reference only",
}


# --------------------------------------------------------------------------
# Noise metrics, read back off the per-frame CSV
# --------------------------------------------------------------------------


def _rows_by_track(csv_path: Path) -> Dict[int, List[dict]]:
    by: Dict[int, List[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by[int(row["track_id"])].append(row)
    return by


def noise_metrics(csv_path: Path, stationary_speed_in_s: float) -> Dict[str, Any]:
    by = _rows_by_track(csv_path)

    jumps: List[float] = []
    peak = 0.0
    rest_steps: List[float] = []
    short: List[int] = []

    for track_id, rows in sorted(by.items()):
        speeds = [float(r["speed_in_s"]) for r in rows]
        peak = max(peak, max(speeds, default=0.0))
        if len(rows) < 15:
            short.append(track_id)

        # Speed jitter, over the stretches where the ball is actually moving.
        # A ball at rest has a speed estimate of ~0 that cannot jitter, so
        # including it would make a clip look better the more of it was idle.
        for i in range(1, len(speeds)):
            if max(speeds[i], speeds[i - 1]) > 2.5 * stationary_speed_in_s:
                jumps.append(abs(speeds[i] - speeds[i - 1]))

        # Position jitter, over the stretches where it is not.
        for i in range(1, len(rows)):
            if max(speeds[i], speeds[i - 1]) > stationary_speed_in_s:
                continue
            step = math.hypot(
                float(rows[i]["x_in"]) - float(rows[i - 1]["x_in"]),
                float(rows[i]["y_in"]) - float(rows[i - 1]["y_in"]),
            )
            rest_steps.append(step)

    return {
        "tracks_reported": len(by),
        "short_tracks": short,
        "peak_speed_in_s": round(peak, 1),
        "speed_jitter_in_s": {
            "mean": round(sum(jumps) / len(jumps), 2) if jumps else None,
            "max": round(max(jumps), 1) if jumps else None,
            "samples": len(jumps),
        },
        "rest_jitter_in": {
            "mean": round(sum(rest_steps) / len(rest_steps), 3) if rest_steps else None,
            "max": round(max(rest_steps), 3) if rest_steps else None,
            "samples": len(rest_steps),
        },
    }


# --------------------------------------------------------------------------
# One clip
# --------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise (``--out-dir``)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def measure_clip(
    name: str, spec: Dict[str, Any], out_dir: Path, render: bool = True
) -> Dict[str, Any]:
    video = ROOT / spec["video"]
    if not video.exists():
        return {"skipped": f"{spec['video']} is not in the repo"}

    cfg = Config().apply_preset()
    csv_path = out_dir / f"{name}_tracks.csv"
    json_path = out_dir / f"{name}_run.json"
    video_path = out_dir / f"{name}_tracked.mp4"
    preview_path = out_dir / f"preview_{name}.png"

    summary = run(
        cfg,
        RunOptions(
            video=str(video),
            output=str(video_path) if render else None,
            export_csv=str(csv_path),
            export_json=str(json_path),
            progress_every=0,
        ),
    )
    written = [csv_path, json_path] + ([video_path] if render else [])

    if render:
        # The preview panel lives in the CLI's calibrate command, and shelling
        # out to it is also what produced the file that is already in results/,
        # so the two stay comparable.
        proc = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "calibrate", str(video),
             "--save-preview", str(preview_path)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            written.append(preview_path)

    record: Dict[str, Any] = {
        "written": [_display_path(p) for p in written],
        "balls_visible": spec["balls_visible"],
        "real_pots": spec["real_pots"],
        "detectable_pots": spec["detectable_pots"],
        "frames": summary["frames_processed"],
        # ``.get``, so the tool can also be pointed at an older checkout to
        # produce the "before" entry for a comparison.
        "frames_repeated": summary.get("frames_repeated"),
        "clock": summary.get("clock"),
        "frames_view_lost": summary["frames_view_lost"],
        "recalibrations": summary["recalibrations"],
        "tracks_created": summary["tracks_created"],
        "events": summary["events"],
        "shots": [s["summary"] for s in summary["shot_log"]],
        "processing_fps": summary["processing_fps"],
    }
    record.update(noise_metrics(csv_path, cfg.tracker.stationary_speed_in_s))
    return record


#: The second synthetic clip: the same break, captured the way the sample
#: broadcasts evidently were -- 25 fps content grabbed by a 37.5 fps recorder
#: that repeats frames, receives them late and misses about one in eight.
#: Its ground truth is what shows whether the tracker's clock is right.
RETIMED_CAPTURE = ["--fps", "25", "--container-fps", "37.5",
                   "--drop-rate", "0.12", "--capture-jitter", "0.6"]


def measure_ground_truth(
    out_dir: Path, render: bool = True, retimed: bool = False
) -> Dict[str, Any]:
    """The accuracy side, so a noise fix that cost accuracy cannot hide."""
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import (  # noqa: E402
        evaluate, evaluate_events, load_ground_truth, load_speeds, load_tracks,
        load_visibility,
    )

    stem = "synthetic_retimed" if retimed else "synthetic"
    video = out_dir / f"{stem}_break.mp4"
    gt = out_dir / f"{stem}_gt.csv"
    events_gt = out_dir / f"{stem}_events.json"
    tracks = out_dir / f"{stem}_tracks.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_synthetic_clip.py"),
         "--out", str(video), "--ground-truth", str(gt), "--events", str(events_gt)]
        + (RETIMED_CAPTURE if retimed else []),
        check=True, capture_output=True,
    )

    summary = run(
        Config().apply_preset(),
        RunOptions(
            video=str(video),
            output=str(out_dir / f"{stem}_tracked.mp4") if render else None,
            export_csv=str(tracks),
            progress_every=0,
        ),
    )
    report = evaluate(
        load_ground_truth(gt), load_tracks(tracks), gate_in=2.25,
        gt_speeds=load_speeds(gt, "ball"), track_speeds=load_speeds(tracks, "track_id"),
        gt_visibility=load_visibility(gt),
    )
    events = evaluate_events(
        json.loads(events_gt.read_text(encoding="utf-8")),
        summary["event_log"],
        fps=float(summary["video"]["fps"]),
    )
    speed = report.get("speed_error") or {}
    return {
        "recall": round(report["recall"], 4),
        "precision": round(report["precision"], 4),
        "mota": round(report["mota"], 4),
        "id_switches": report["id_switches"],
        "position_error_in_median": round(report["position_error_in"]["median"], 3),
        "speed_error_median_relative": speed.get("median_relative"),
        "speed_error_p95_in_s": speed.get("p95_in_s"),
        "events_found": {
            kind: f"{e['matched']}/{e['true']} (+{e['false_positives']} false)"
            for kind, e in events.items()
        },
        "clock": summary.get("clock"),
    }


# --------------------------------------------------------------------------
# The log
# --------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(ROOT)) + args,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git may not be installed
        return ""


def load_log() -> Dict[str, Any]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    return {
        "schema": 1,
        "what_this_is": (
            "One entry per 'changed something, re-measured it' cycle on the "
            "sample clips.  Read the newest entry to see where the tracker "
            "stands and what is still wrong; read note/open_issues to see why "
            "the numbers moved.  Regenerate with tools/run_report.py."
        ),
        "how_to_regenerate": "python tools/run_report.py --note '<what changed>'",
        "metrics": METRICS_MEANING,
        "clips": SAMPLE_CLIPS,
        "runs": [],
    }


def amend(log: Dict[str, Any], issues: Sequence[str]) -> None:
    """Correct the newest entry's *metadata* without re-running anything.

    For what a closer look at the clips turned up: extra open issues, and a
    ``balls_visible`` or ``real_pots`` that was wrong when the entry was
    written.  It never touches a measured number -- those only ever come from
    a run.
    """
    if not log["runs"]:
        raise SystemExit("nothing to amend: the log is empty")
    # What is in a clip is a property of the clip, not of the run that looked
    # at it, so a correction applies to every entry that ever measured it.
    for entry in log["runs"]:
        for name, clip in entry["clips"].items():
            spec = SAMPLE_CLIPS.get(name)
            if spec and "skipped" not in clip:
                clip["balls_visible"] = spec["balls_visible"]
                clip["real_pots"] = spec["real_pots"]
                clip["detectable_pots"] = spec["detectable_pots"]

    entry = log["runs"][-1]
    for issue in issues:
        if issue not in entry["open_issues"]:
            entry["open_issues"].append(issue)


def save_log(log: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def print_log(log: Dict[str, Any]) -> None:
    for entry in log["runs"]:
        print(f"\n=== {entry['id']}  ({entry['commit'] or '?'}{', dirty tree' if entry.get('dirty') else ''})")
        print(f"    {entry['note']}")
        for key, title in (("ground_truth", "ground truth"),
                           ("ground_truth_retimed", "retimed truth")):
            gt = entry.get(key)
            if not gt:
                continue
            line = (
                f"    {title}: recall {gt['recall']} / precision {gt['precision']} / "
                f"MOTA {gt['mota']} / {gt['id_switches']} id switches"
            )
            if gt.get("speed_error_median_relative") is not None:
                line += (
                    f" / speed error {100 * gt['speed_error_median_relative']:.1f}%"
                    f" (p95 {gt['speed_error_p95_in_s']} in/s)"
                )
            print(line)
            if gt.get("events_found"):
                print("      events found: " + ", ".join(
                    f"{k} {v}" for k, v in gt["events_found"].items()))
        for name, r in (entry.get("real") or {}).items():
            print(f"    real {name:<16} score {r['score']} / found {r['found']} / real {r['real']} / "
                  f"named {r['named_right']} (wrong {r['named_wrong']}) / {r['ids_per_ball']} ids per ball / "
                  f"{r['phantoms_per_keyframe']} phantoms per keyframe")
        for name, clip in entry["clips"].items():
            if "skipped" in clip:
                print(f"    {name:<14} skipped: {clip['skipped']}")
                continue
            jitter = clip["speed_jitter_in_s"]
            print(
                f"    {name:<14} {clip['tracks_reported']:>3} tracks reported "
                f"({clip['tracks_created']} created) for "
                f"{clip['balls_visible']} balls, "
                f"peak {clip['peak_speed_in_s']:>5} in/s, "
                f"speed jitter {jitter['mean']}, "
                f"rest jitter {clip['rest_jitter_in']['mean']}, "
                f"short {clip['short_tracks']}"
            )
            pots = clip["events"].get("pot", 0)
            real = clip.get("detectable_pots", clip.get("real_pots"))
            flag = "" if real is None or pots == real else f"  (detectable pots: {real})"
            print(f"    {'':<14} {clip['events']}{flag}  {clip['shots']}")
        for issue in entry.get("open_issues", []):
            print(f"      - still open: {issue}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--note", help="what changed since the last entry")
    ap.add_argument(
        "--open-issue",
        action="append",
        default=[],
        dest="open_issues",
        help="something this run did not fix (repeatable)",
    )
    ap.add_argument(
        "--clips",
        nargs="*",
        choices=sorted(SAMPLE_CLIPS),
        help="only these clips (default: all of them)",
    )
    ap.add_argument(
        "--ground-truth",
        action="store_true",
        help="also build the synthetic clip and score accuracy (slow)",
    )
    ap.add_argument(
        "--real",
        action="store_true",
        help="also score real footage against the answer keys in tools/truth/ "
             "(tools/real_eval.py; needs the clips in .cache/)",
    )
    ap.add_argument(
        "--no-render",
        action="store_true",
        help="skip the annotated video and the calibration preview; the "
             "numbers are the same and the run is quicker, but there is then "
             "nothing new in results/ to look at",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results"),
        help="where the annotated videos, CSVs, JSON and previews go "
             "(results/, which is not committed)",
    )
    ap.add_argument("--show", action="store_true", help="print the log and exit")
    ap.add_argument(
        "--amend",
        action="store_true",
        help="add --open-issue notes to the newest entry and refresh the "
             "per-clip ground truth, without re-running the clips",
    )
    args = ap.parse_args(argv)

    log = load_log()
    if args.show:
        print_log(log)
        return 0
    if args.amend:
        log["clips"] = SAMPLE_CLIPS
        amend(log, args.open_issues)
        save_log(log)
        print_log({"runs": log["runs"][-1:]})
        return 0
    if not args.note:
        ap.error("--note is required: say what changed, for the next session")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render = not args.no_render
    names = args.clips or sorted(SAMPLE_CLIPS)
    clips: Dict[str, Any] = {}
    for name in names:
        print(f"[report] {name} ...", flush=True)
        clips[name] = measure_clip(name, SAMPLE_CLIPS[name], out_dir, render)

    entry: Dict[str, Any] = {
        "id": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "note": args.note,
        "clips": clips,
        "open_issues": list(args.open_issues),
    }
    if args.ground_truth:
        print("[report] synthetic ground truth ...", flush=True)
        entry["ground_truth"] = measure_ground_truth(out_dir, render)
        print("[report] synthetic ground truth, retimed like a broadcast ...", flush=True)
        entry["ground_truth_retimed"] = measure_ground_truth(out_dir, render, retimed=True)

    if args.real:
        sys.path.insert(0, str(ROOT / "tools"))
        import real_eval

        entry["real"] = {}
        for truth in real_eval.all_truths():
            print(f"[report] real footage: {truth['name']} ...", flush=True)
            got = real_eval.measure(truth, ROOT, out_dir / "real", render=render)
            entry["real"][truth["name"]] = {k: v for k, v in got.items() if k not in ("name", "per_keyframe")}

    log.setdefault("metrics", METRICS_MEANING).update(METRICS_MEANING)
    log["runs"].append(entry)
    save_log(log)
    print_log({"runs": [entry]})
    print(f"\n[report] appended to {LOG_PATH.relative_to(ROOT)}")

    # Name the files, with their timestamps: the whole point of rewriting them
    # is that someone opens them, and a path they cannot tell is fresh is no
    # better than the stale one it replaced.
    written = [w for clip in clips.values() for w in clip.get("written", [])]
    if written:
        print("[report] rewrote, watch these:")
        for path in sorted(written, key=lambda p: (not p.endswith(".mp4"), p)):
            full = Path(path) if Path(path).is_absolute() else ROOT / path
            stamp = datetime.fromtimestamp(full.stat().st_mtime).strftime("%H:%M:%S")
            print(f"           {stamp}  {path}")
    elif render:
        print("[report] nothing was written to results/")
    else:
        print("[report] --no-render: results/ still holds the previous run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
