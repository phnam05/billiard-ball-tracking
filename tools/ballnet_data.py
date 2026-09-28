#!/usr/bin/env python
"""Training data for the ball model (``billiards/ballnet.py``).

The model is shown a small picture centred on each ball the detector
proposes and says whether it is a ball, the cue ball, and which colour and
pattern it is.  Its examples come from two places:

* **real footage**: the clips in ``REAL_CLIPS``, never the ones the answer
  keys in ``tools/truth/`` were marked on, so those stay an honest test.
  Each clip is tracked, a crop of every proposal is kept with the track it
  went to, and the tracks are then labelled by eye from contact sheets
  (``sheets``): a ball and its colour, or not a ball.  The labels are kept
  in ``tools/ballnet/labels/<clip>.json`` by track and turned into rows of
  ``(clip, frame, x, y, radius, label)`` (``tools/ballnet/real.csv``), so the
  crops can be cut again from the clips without re-tracking them;
* **the simulator** (``tools/make_synthetic_clip.py``), whose every ball is
  known, over cloths, cameras, floors and both ball sets.

Usage::

    python tools/ballnet_data.py harvest mosconi uk-open ...   # track, keep crops
    python tools/ballnet_data.py sheets mosconi                # sheets to label from
    python tools/ballnet_data.py rows                          # labels -> real.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from billiards import ballnet  # noqa: E402

DATA = ROOT / "tools" / "ballnet"
WORK = ROOT / "tmp" / "ballnet"

#: Crops are stored this many pixels square; the model sees them at
#: ``billiards.ballnet.INPUT`` (training shifts and scales them first).
CROP_RADII = ballnet.CROP_RADII
STORE_PX = 48

#: The colour families of ``billiards.balls.FAMILIES``, plus black.
FAMILIES = ["yellow", "blue", "red", "pink", "purple", "orange", "green", "maroon", "black"]

#: Labels: "no" (not a ball), "cue", a family ("red"), a family with
#: "-stripe" ("red-stripe"), or "ball" (a ball whose colour cannot be told).
LABELS = ["no", "cue", "ball"] + FAMILIES + [f + "-stripe" for f in FAMILIES if f != "black"]


#: Training clips: every cached venue minute and CCTV-style clip except the
#: ones with answer keys (us-open, premier-league, demo-table-cam,
#: county-8ball) and the us-open venue minute, which is the same match; and
#: the three sample clips, whose Aramith TV set is the commonest tournament
#: set (without them the model had never seen its dark blue 2, and dropped
#: it on the 2026 Premier League final).  (name, file, preset, numbers)
REAL_CLIPS: Dict[str, Tuple[str, str, str]] = {
    "albin_fedor": ("albin_fedor.mp4", "pool-9ft", "1-15"),
    "fedor_shot": ("fedor_shot.mp4", "pool-9ft", "1-15"),
    "fedor_jump": ("fedor_jump.mp4", "pool-9ft", "1-15"),
    "wpa-2026": (".cache/venues/wpa-2026.mp4", "pool-9ft", "1-10"),
    "mosconi": (".cache/venues/mosconi.mp4", "pool-9ft", "1-9"),
    "uk-open": (".cache/venues/uk-open.mp4", "pool-9ft", "1-9"),
    "hanoi-open": (".cache/venues/hanoi-open.mp4", "pool-9ft", "1-9"),
    "heyball": (".cache/venues/heyball.mp4", "pool-9ft", "1-15"),
    "derby-city": (".cache/venues/derby-city.mp4", "pool-9ft", "1-9"),
    "bar-box": (".cache/venues/bar-box.mp4", "pool-7ft", "1-15"),
    "snooker": (".cache/venues/snooker.mp4", "snooker-12ft", "1-15"),
    "apa-mallett": (".cache/cctv/apa-mallett.mp4", "pool-9ft", "1-15"),
    "bca-brad-ivan": (".cache/cctv/bca-brad-ivan.mp4", "pool-9ft", "1-9"),
    "overhead-home": (".cache/cctv/overhead-home.mp4", "pool-7ft", "1-15"),
}
# ghost-pool and live-pool-cam were tried too: no table was found in either.

#: Crops kept per track: spread over its life.  Short tracks keep them all.
PER_TRACK = 64


def crop(frame: np.ndarray, x: float, y: float, r: float, size: int = STORE_PX) -> np.ndarray:
    """The square ``CROP_RADII`` ball radii wide round (x, y): as the tracker
    cuts it for the model (``billiards.ballnet.crop``), at ``size`` pixels."""
    return ballnet.crop(frame, x, y, r, size)


# --------------------------------------------------------------------------
# Real footage
# --------------------------------------------------------------------------


def harvest(name: str) -> Path:
    """Track one clip; keep every proposal and a sample of crops."""
    from billiards.app.workspace import build_config, clean_settings
    from billiards.pipeline import RunOptions, build_pipeline
    from billiards.video import read_frames

    file, preset, numbers = REAL_CLIPS[name]
    video = ROOT / file
    cfg = build_config(clean_settings({"preset": preset, "numbers": numbers, "far_cushion": True}))
    pipe, _, info = build_pipeline(cfg, RunOptions(video=str(video)))
    tracker = pipe.tracker
    props: List[Tuple[int, float, float, float, int, int]] = []  # frame, x, y, r, track, cluster
    current = {"frame": 0}
    original = tracker.update

    def update(detections, dt, frame, t_s):
        out = original(detections, dt, frame, t_s)
        seen = {}
        for t in tracker.tracks:
            if t.time_since_update == 0:
                seen[tuple(np.round(t.last_observed_xy, 4))] = t.track_id
        for d in detections:
            tid = seen.get(tuple(np.round(d.centre_table, 4)), -1)
            props.append((frame, d.centre_image[0], d.centre_image[1], d.radius_px, tid, int(d.from_cluster)))
        return out

    tracker.update = update
    for idx, frame in read_frames(str(video), 0, None, cfg.max_frame_width):
        current["frame"] = idx
        pipe.process(frame, idx, annotate=False)
    pipe.finish()

    tracks = {t.track_id: t for t in tracker.all_tracks()}
    by_track: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(props):
        by_track[p[4]].append(i)
    keep: List[int] = []
    for tid, idxs in by_track.items():
        if len(idxs) <= PER_TRACK:
            keep.extend(idxs)
        else:
            keep.extend(idxs[int(k)] for k in np.linspace(0, len(idxs) - 1, PER_TRACK))
    keep.sort()
    want: Dict[int, List[int]] = defaultdict(list)
    for i in keep:
        want[props[i][0]].append(i)
    crops = np.zeros((len(keep), STORE_PX, STORE_PX, 3), np.uint8)
    slot = {i: k for k, i in enumerate(keep)}
    for idx, frame in read_frames(str(video), 0, None, cfg.max_frame_width):
        for i in want.get(idx, []):
            _, x, y, r, _, _ = props[i]
            crops[slot[i]] = crop(frame, x, y, r)

    WORK.mkdir(parents=True, exist_ok=True)
    meta = {
        "clip": name, "file": file, "width": cfg.max_frame_width,
        "rows": [list(map(lambda v: round(float(v), 2) if isinstance(v, float) else v, props[i])) for i in keep],
        "tracks": {
            str(tid): {
                "proposals": len(idxs),
                "state": tracks[tid].state.value if tid in tracks else "none",
                "death": tracks[tid].death_reason if tid in tracks else None,
                "label": tracks[tid].label if tid in tracks else "",
                "first": props[idxs[0]][0], "last": props[idxs[-1]][0],
            }
            for tid, idxs in by_track.items()
        },
    }
    out = WORK / f"{name}.npz"
    np.savez_compressed(out, crops=crops)
    (WORK / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[harvest] {name}: {len(props)} proposals, {len(by_track)} tracks, {len(keep)} crops kept")
    return out


def sheets(name: str, per_row: int = 8, rows_per_sheet: int = 12, only: Optional[Sequence[int]] = None) -> List[Path]:
    """Contact sheets: one row per track, ``per_row`` of its crops, labelled."""
    meta = json.loads((WORK / f"{name}.json").read_text(encoding="utf-8"))
    crops = np.load(WORK / f"{name}.npz")["crops"]
    by_track: Dict[int, List[int]] = defaultdict(list)
    for k, row in enumerate(meta["rows"]):
        by_track[int(row[4])].append(k)
    tids = sorted(by_track, key=lambda t: (-len(by_track[t]), t))
    if only:
        tids = [t for t in tids if t in set(only)]
    tile = 64
    out: List[Path] = []
    # Tracks of a few proposals -- nearly all of them never confirmed -- go on
    # grids of one crop each; the rest get a row of crops over their life.
    short = [t for t in tids if len(by_track[t]) <= 3]
    tids = [t for t in tids if len(by_track[t]) > 3]
    cols, per_grid = 12, 120
    for s in range(0, len(short), per_grid):
        cells = []
        for tid in short[s:s + per_grid]:
            k = by_track[tid][len(by_track[tid]) // 2]
            cell = cv2.resize(crops[k], (tile, tile), interpolation=cv2.INTER_CUBIC)
            cv2.putText(cell, str(tid), (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cells.append(cell)
        cells += [np.zeros((tile, tile, 3), np.uint8)] * (-len(cells) % cols)
        grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        path = WORK / f"sheet_{name}_short_{s // per_grid:02d}.jpg"
        cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(path)
    for s in range(0, len(tids), rows_per_sheet):
        lines = []
        for tid in tids[s:s + rows_per_sheet]:
            idxs = by_track[tid]
            pick = [idxs[int(k)] for k in np.linspace(0, len(idxs) - 1, min(per_row, len(idxs)))]
            cells = [cv2.resize(crops[k], (tile, tile), interpolation=cv2.INTER_CUBIC) for k in pick]
            cells += [np.zeros((tile, tile, 3), np.uint8)] * (per_row - len(cells))
            info = meta["tracks"].get(str(tid), {})
            head = np.zeros((tile, 150, 3), np.uint8)
            cv2.putText(head, f"T{tid}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(head, f"{info.get('proposals', 0)}p {info.get('label', '')}", (4, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(head, f"{info.get('death') or info.get('state', '')} f{info.get('first')}", (4, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            lines.append(np.hstack([head] + cells))
        path = WORK / f"sheet_{name}_{s // rows_per_sheet:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(path)
    print(f"[sheets] {name}: {len(tids)} tracks, {len(out)} sheets in {WORK.relative_to(ROOT)}")
    return out


def rows() -> Path:
    """Track labels (``tools/ballnet/labels/*.json``) -> ``tools/ballnet/real.csv``.

    A label file maps a track id to a label, and may give ``"default"`` for
    every track not named and ``"frames"`` overrides: ``{"12": {"<=400":
    "red", ">400": "no"}}`` for a track that changed ball.
    """
    out = DATA / "real.csv"
    n = defaultdict(int)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "file", "width", "frame", "x", "y", "r", "label", "track"])
        for path in sorted((DATA / "labels").glob("*.json")):
            name = path.stem
            labels = json.loads(path.read_text(encoding="utf-8"))
            meta = json.loads((WORK / f"{name}.json").read_text(encoding="utf-8"))
            default = labels.get("default")
            for frame, x, y, r, tid, _ in meta["rows"]:
                label = labels.get(str(tid), default)
                if isinstance(label, dict):
                    label = next((v for k, v in label.items()
                                  if (k.startswith("<=") and frame <= int(k[2:]))
                                  or (k.startswith(">") and frame > int(k[1:]))), None)
                if label is None or label == "skip":
                    continue
                assert label in LABELS, (name, tid, label)
                w.writerow([name, meta["file"], meta["width"], frame, x, y, r, label, tid])
                n[label] += 1
    print(f"[rows] wrote {out.relative_to(ROOT)}: {sum(n.values())} crops, " +
          ", ".join(f"{k} {v}" for k, v in sorted(n.items(), key=lambda kv: -kv[1])))
    return out


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


def synthetic(scenes: int = 48, seed: int = 0, frames_per_scene: int = 36) -> Path:
    """Crops of simulated balls, and of what is round them, with exact labels.

    Each scene is a break filmed from one camera, on one cloth and floor, at
    one picture size, with nine balls drawn at random from one set: the
    simulator's two racks alone would never show a red stripe or a green
    one.  Every ball at least 0.6 visible is a crop; so are points on the
    cloth, the rails, the pockets and the floor, which are not balls.
    """
    import make_synthetic_clip as S

    rng = np.random.default_rng(seed)
    cams = list(S.CAMERAS)
    cloths = list(S.CLOTHS)
    floors = list(S.FLOORS)
    sizes = [(1280, 720), (960, 540), (1280, 960)]
    out_crops: List[np.ndarray] = []
    out_labels: List[str] = []
    out_scene: List[int] = []
    for k in range(scenes):
        ball_set = "tv" if k % 2 else "standard"
        cam, cloth, floor = cams[k % len(cams)], cloths[(k // 2) % len(cloths)], floors[(k // 3) % len(floors)]
        size = sizes[k % len(sizes)]
        balls = S.make_break_rack(seed + k, "tv" if ball_set == "tv" else "standard")
        # Nine of the fifteen, in this set's colours.
        numbers = sorted(rng.choice(np.arange(1, 16), size=len(balls) - 1, replace=False))
        colour_of = {1: "yellow", 2: "blue", 3: "red", 4: "pink" if ball_set == "tv" else "purple",
                     5: "purple" if ball_set == "tv" else "orange", 6: "green", 7: "maroon", 8: "black"}
        for b, n in zip(balls[1:], numbers):
            n = int(n)
            fam = colour_of[n if n <= 8 else n - 8]
            b.number, b.striped = n, n > 8
            b.colour, b.caps = S.BALL_COLOURS[fam], S.CAPS[ball_set]
            b.name = fam + ("-stripe" if n > 8 else "")
        renderer = S.Renderer(out_size=size, seed=seed + k, camera=cam, cloth_bgr=S.CLOTHS[cloth],
                              floor_bgr=S.FLOORS[floor])
        sim = S.Simulation(balls, seed=seed + k)
        aim = np.array([1.0, rng.normal(0, 0.05)])
        balls[0].vel = aim / np.linalg.norm(aim) * float(rng.uniform(150, 320))
        dt = 1.0 / 30.0
        for f in range(frames_per_scene * 5):
            sim.step(dt)
            if f % 5:
                continue
            frame = renderer.render(balls)
            seen = renderer.visibility(balls)
            placed = []
            for b in balls:
                if not b.active:
                    continue
                x, y = renderer.ball_to_image(b.pos)
                r = renderer.sphere_radius_px(b.pos)
                placed.append((x, y, r))
                if seen[b.name] < 0.6 or not (0 <= x < size[0] and 0 <= y < size[1]):
                    continue
                # A little off-centre, as a detector's centre is.
                jx, jy = rng.normal(0, 0.12 * r, 2)
                out_crops.append(crop(frame, x + jx, y + jy, r))
                out_labels.append(b.name)
                out_scene.append(k)
            # Not balls: points on and round the table, away from every ball.
            for _ in range(6):
                p = (rng.uniform(-6, S.TABLE_L + 6), rng.uniform(-6, S.TABLE_W + 6))
                inside = (min(max(p[0], 1.2), S.TABLE_L - 1.2), min(max(p[1], 1.2), S.TABLE_W - 1.2))
                x, y = renderer.ball_to_image(p)
                r = renderer.sphere_radius_px(inside) * rng.uniform(0.8, 1.25)
                if not (0 <= x < size[0] and 0 <= y < size[1]):
                    continue
                if any((x - bx) ** 2 + (y - by) ** 2 < (1.6 * (r + br)) ** 2 for bx, by, br in placed):
                    continue
                out_crops.append(crop(frame, x, y, r))
                out_labels.append("no")
                out_scene.append(k)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "synthetic.npz"
    np.savez_compressed(path, crops=np.array(out_crops, np.uint8), labels=np.array(out_labels),
                        scene=np.array(out_scene))
    counts = defaultdict(int)
    for lab in out_labels:
        counts[lab] += 1
    print(f"[synthetic] {len(out_crops)} crops from {scenes} scenes: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("clips", nargs="*", default=sorted(REAL_CLIPS))
    s = sub.add_parser("sheets")
    s.add_argument("clips", nargs="+")
    s.add_argument("--tracks", nargs="*", type=int)
    sub.add_parser("rows")
    y = sub.add_parser("synthetic")
    y.add_argument("--scenes", type=int, default=48)
    a = ap.parse_args(argv)
    if a.cmd == "harvest":
        for c in a.clips:
            harvest(c)
    elif a.cmd == "sheets":
        for c in a.clips:
            sheets(c, only=a.tracks)
    elif a.cmd == "synthetic":
        synthetic(a.scenes)
    else:
        rows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
