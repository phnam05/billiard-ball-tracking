#!/usr/bin/env python
"""Train the ball model (``billiards/ballnet.py``) and export it for OpenCV.

    python tools/train_ballnet.py                 # train, write billiards/models/
    python tools/train_ballnet.py --epochs 3      # a quick look

Needs PyTorch (CPU is enough: a few minutes) and ``onnx``; the tracker does
not -- it runs the exported ``ballnet.onnx`` through OpenCV.  The examples are
``tools/ballnet/real.csv`` (crops of real clips, labelled by eye; the crops
are cut from the clips again, which must be in ``.cache/``) and the
simulator's (``tools/ballnet_data.py synthetic``, made again if missing).

Held out, for the numbers in the model's JSON: one real track in five, from
every clip (whole tracks, so near-identical crops of one ball are never on
both sides).  What is exported is the running average of the weights at the
end of training.  The answer keys' clips were never in the data at all;
``tools/real_eval.py`` is the test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from billiards.ballnet import CROP_RADII, FAMILIES, INPUT, KINDS, MODEL_DIR, crop  # noqa: E402

import ballnet_data  # noqa: E402

#: One track in this many is held out.
VALIDATION_EVERY = 5
#: Weight of the running average of the weights, per step.
_EMA = 0.998
STORE = ballnet_data.STORE_PX


def encode(label: str) -> Tuple[int, int, int]:
    """(kind, family, stripe) for a label; -1 where it says nothing."""
    if label == "no":
        return 0, -1, -1
    if label == "cue":
        return 1, -1, -1
    if label == "ball":
        return 2, -1, -1
    striped = label.endswith("-stripe")
    family = label[: -len("-stripe")] if striped else label
    return 2, FAMILIES.index(family), int(striped)


def real_crops() -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Crops of every row of ``tools/ballnet/real.csv``, cut from the clips."""
    path = ballnet_data.DATA / "real.csv"
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    cache = ballnet_data.WORK / f"real_crops_{digest}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["crops"], list(d["labels"]), list(d["clips"]), list(d["tracks"])
    from billiards.video import read_frames

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_file: Dict[Tuple[str, int], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_file[(r["file"], int(r["width"]))][int(r["frame"])].append(i)
    crops = np.zeros((len(rows), STORE, STORE, 3), np.uint8)
    for (file, width), frames in by_file.items():
        last = max(frames)
        for idx, frame in read_frames(str(ROOT / file), 0, last + 1, width):
            for i in frames.get(idx, []):
                r = rows[i]
                crops[i] = crop(frame, float(r["x"]), float(r["y"]), float(r["r"]), STORE)
    labels = [r["label"] for r in rows]
    clips = [r["clip"] for r in rows]
    tracks = [r["track"] for r in rows]
    np.savez_compressed(cache, crops=crops, labels=np.array(labels), clips=np.array(clips),
                        tracks=np.array(tracks))
    return crops, labels, clips, tracks


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random scale and shift (a detector's centre and size are not exact),
    flips and quarter turns (a ball has no up), and the camera's colour,
    brightness, blur and noise.  ``batch``: (n, STORE, STORE, 3) uint8."""
    n = len(batch)
    out = np.empty((n, INPUT, INPUT, 3), np.float32)
    for i in range(n):
        img = batch[i]
        # From a crop filled by part of a ball (the detector's radius was too
        # small: a CCTV-style tripod camera's were half) to one it is small in.
        s = STORE * rng.uniform(0.62, 1.45)
        cx = STORE / 2 + rng.normal(0, 0.05 * STORE)
        cy = STORE / 2 + rng.normal(0, 0.05 * STORE)
        src = np.float32([[cx - s / 2, cy - s / 2], [cx + s / 2, cy - s / 2], [cx - s / 2, cy + s / 2]])
        dst = np.float32([[0, 0], [INPUT, 0], [0, INPUT]])
        img = cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (INPUT, INPUT),
                             flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        img = np.rot90(img, int(rng.integers(4)))
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))
        f = img.astype(np.float32)
        # Colour: a camera's white balance and saturation, gently -- a big
        # hue shift would turn a red ball into an orange one.
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.normal(0, 2.5)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        f = f * rng.uniform(0.75, 1.25) + rng.normal(0, 12)
        f = f * rng.uniform(0.92, 1.08, 3)[None, None, :]
        f += rng.normal(0, rng.uniform(0, 5), f.shape)
        out[i] = f
    return np.clip(out, 0, 255)


def to_input(batch: np.ndarray) -> np.ndarray:
    """(n, INPUT, INPUT, 3) 0-255 -> (n, 3, INPUT, INPUT), as ``BallNet.score`` feeds it."""
    return ((batch - 127.5) / 255.0).transpose(0, 3, 1, 2).astype(np.float32)


def centre(batch: np.ndarray) -> np.ndarray:
    """The stored crops as the tracker cuts them: no augmentation."""
    return np.stack([cv2.resize(b, (INPUT, INPUT), interpolation=cv2.INTER_AREA) for b in batch]).astype(np.float32)


def build_model():
    import torch.nn as nn

    def block(a, b):
        return [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                *block(3, 16), *block(16, 16), nn.MaxPool2d(2),
                *block(16, 32), *block(32, 32), nn.MaxPool2d(2),
                *block(32, 64), *block(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.kind = nn.Linear(64, len(KINDS))
            self.family = nn.Linear(64, len(FAMILIES))
            self.stripe = nn.Linear(64, 1)

        def forward(self, x):
            h = self.body(x)
            return self.kind(h), self.family(h), self.stripe(h).squeeze(1)

    return Net()


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    import torch

    model.eval()
    kinds, fams, strs = [], [], []
    with torch.no_grad():
        for s in range(0, len(x), 1024):
            k, f, st = model(torch.from_numpy(to_input(x[s:s + 1024])))
            kinds.append(k.argmax(1).numpy()); fams.append(f.argmax(1).numpy()); strs.append((st > 0).numpy())
    k, f, st = np.concatenate(kinds), np.concatenate(fams), np.concatenate(strs)
    out: Dict[str, float] = {}
    ball = y[:, 0] > 0
    notball = y[:, 0] == 0
    if ball.any():
        out["balls_kept"] = round(float(np.mean(k[ball] > 0)), 4)
    if notball.any():
        out["non_balls_rejected"] = round(float(np.mean(k[notball] == 0)), 4)
    cue = y[:, 0] == 1
    if cue.any():
        out["cue_right"] = round(float(np.mean(k[cue] == 1)), 4)
    fam = y[:, 1] >= 0
    if fam.any():
        out["family_right"] = round(float(np.mean(f[fam] == y[fam, 1])), 4)
    s = y[:, 2] >= 0
    if s.any():
        out["stripe_right"] = round(float(np.mean(st[s] == y[s, 2])), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--out", default=str(MODEL_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    syn_path = ballnet_data.WORK / "synthetic.npz"
    if not syn_path.exists():
        ballnet_data.synthetic()
    syn = np.load(syn_path)
    sx, sl = syn["crops"], list(syn["labels"])
    rx, rl, rc, rt = real_crops()
    val = np.array([int(hashlib.sha1(f"{c}/{t}".encode()).hexdigest(), 16) % VALIDATION_EVERY == 0
                    for c, t in zip(rc, rt)])

    def labels(ls: List[str]) -> np.ndarray:
        return np.array([encode(l) for l in ls], dtype=np.int64).reshape(-1, 3)

    xs = np.concatenate([sx, rx[~val]])
    ys = np.concatenate([labels(sl), labels([l for l, v in zip(rl, val) if not v])])
    source = np.concatenate([np.zeros(len(sx), int), np.ones(int((~val).sum()), int)])
    vx, vy = centre(rx[val]), labels([l for l, v in zip(rl, val) if v])
    print(f"[train] {len(sx)} synthetic + {int((~val).sum())} real crops; validation {len(vx)} real "
          f"(one track in {VALIDATION_EVERY})")
    print("[train] real labels:", dict(Counter(rl).most_common()))

    # Each batch half real, half simulated, and half balls, half not: the
    # real crops are what matters, and they are fewer.
    groups = {(s, b): np.nonzero((source == s) & ((ys[:, 0] > 0) == b))[0] for s in (0, 1) for b in (False, True)}
    groups = {k: v for k, v in groups.items() if len(v)}
    model = build_model()
    # What is evaluated and exported is a running average of the weights: the
    # scores on the held-out clips swung by 30 points from epoch to epoch
    # without it.
    import copy

    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps_per_epoch = max(1, len(xs) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * steps_per_epoch)
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        for _ in range(steps_per_epoch):
            idx = np.concatenate([rng.choice(v, 256 // len(groups)) for v in groups.values()])
            xb = torch.from_numpy(to_input(augment(xs[idx], rng)))
            yb = torch.from_numpy(ys[idx])
            k, f, st = model(xb)
            loss = F.cross_entropy(k, yb[:, 0])
            fam = yb[:, 1] >= 0
            if fam.any():
                loss = loss + F.cross_entropy(f[fam], yb[fam, 1])
            s = yb[:, 2] >= 0
            if s.any():
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(st[s], yb[s, 2].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(_EMA).add_(pm.detach(), alpha=1.0 - _EMA)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            total += float(loss.detach())
        scores = evaluate(ema, vx, vy)
        history.append({"epoch": epoch + 1, "loss": round(total / steps_per_epoch, 4), **scores})
        print(f"[train] epoch {epoch + 1:2d} loss {total / steps_per_epoch:.3f} {scores} ({time.time() - t0:.0f}s)", flush=True)

    model = ema
    model.eval()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "ballnet.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, INPUT, INPUT), str(onnx_path), opset_version=13,
                      input_names=["crop"], output_names=["kind", "family", "stripe"],
                      dynamic_axes={"crop": {0: "n"}, "kind": {0: "n"}, "family": {0: "n"}, "stripe": {0: "n"}},
                      dynamo=False)
    # The same answers from OpenCV as from PyTorch, or the export is wrong.
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    probe = to_input(vx[:16]) if len(vx) else np.zeros((16, 3, INPUT, INPUT), np.float32)
    net.setInput(probe)
    names = net.getUnconnectedOutLayersNames()
    got = dict(zip(names, net.forward(names)))
    with torch.no_grad():
        want = model(torch.from_numpy(probe))
    diff = max(float(np.max(np.abs(np.asarray(got[n]).reshape(w.shape) - w.numpy())))
               for n, w in zip(["kind", "family", "stripe"], want))
    assert diff < 1e-3, f"OpenCV disagrees with PyTorch by {diff}"
    final = evaluate(model, vx, vy)
    meta = {
        "input": INPUT, "crop_radii": CROP_RADII, "kinds": KINDS, "families": FAMILIES,
        "normalise": "(BGR - 127.5) / 255",
        "trained": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "examples": {"synthetic": len(sx), "real": int((~val).sum()), "real_labels": dict(Counter(rl))},
        "held_out": f"one real track in {VALIDATION_EVERY}, every clip",
        "validation": final,
        "history": history,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "opencv_matches_pytorch_to": diff,
    }
    (out / "ballnet.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[train] wrote {onnx_path} ({onnx_path.stat().st_size // 1024} KB), validation {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
