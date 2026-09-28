"""A small learned model that looks at each proposed ball.

The detector finds "round things that are not cloth", and decides by hand-made
tests (size, shape, the colour step at the rim) whether each is a ball.  Those
tests pass a chalk cube on a rail, a knuckle, the shadow in a pocket or a logo
on a shirt often enough to matter: against the answer keys of 28 Sep 2026
(``tools/truth/``) the tracker drew 0.4-1.6 phantoms per keyframe, and it read
the balls' numbers from their colour through a palette measured on one
tournament's broadcast, which named a fifth of the balls on the 2026 US Open
wrongly.

This model is shown a small picture centred on each proposal, three ball
radii across, and answers three questions:

* **kind**: not a ball, the cue ball, or another ball;
* **family**: which colour the ball is (``FAMILIES``), for one that is not
  the cue ball;
* **stripe**: whether it is a stripe.

Which *number* a ball is still comes from ``billiards.balls``, because that
depends on the ball set, which the model is not asked about: this only
replaces the colour measurement with a better one.

The model is trained by ``tools/train_ballnet.py`` on crops from the simulator
and from real clips that are not the answer keys' (``tools/ballnet_data.py``),
and ships as ``billiards/models/ballnet.onnx``.  It runs through OpenCV's DNN
module, so nothing new is needed to use it; without the file (or with
``detector.ball_model: off``) the tracker works as before.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ballnet.onnx"

#: What the model is shown: a square ``CROP_RADII`` ball radii wide, round
#: the proposal's centre, scaled to ``INPUT`` pixels.
CROP_RADII = 3.0
INPUT = 32

KINDS = ["no", "cue", "ball"]
#: Proposals are scored this many at a time.
_BATCH = 16
FAMILIES = ["yellow", "blue", "red", "pink", "purple", "orange", "green", "maroon", "black"]


def crop(frame: np.ndarray, x: float, y: float, r: float, size: int = INPUT) -> np.ndarray:
    """The square ``CROP_RADII`` ball radii wide round (x, y), ``size`` pixels."""
    half = 0.5 * CROP_RADII * max(float(r), 2.0)
    src = np.float32([[x - half, y - half], [x + half, y - half], [x - half, y + half]])
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    M = cv2.getAffineTransform(src, dst)
    flags = cv2.INTER_AREA if 2 * half > size else cv2.INTER_LINEAR
    return cv2.warpAffine(frame, M, (size, size), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class BallNet:
    """The model, loaded once; ``score`` batches every proposal of a frame."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = Path(path)
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        meta_path = self.path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.input = int(self.meta.get("input", INPUT))
        self.outputs = self.net.getUnconnectedOutLayersNames()

    def score(
        self, frame: np.ndarray, centres: Sequence[Tuple[float, float]], radii: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Probabilities for each proposal: kind (n, 3), family (n, 9), stripe (n,)."""
        n = len(centres)
        if n == 0:
            return np.zeros((0, 3)), np.zeros((0, len(FAMILIES))), np.zeros(0)
        crops = np.stack([crop(frame, x, y, r, self.input) for (x, y), r in zip(centres, radii)])
        blob = ((crops.astype(np.float32) - 127.5) / 255.0).transpose(0, 3, 1, 2)
        # Always a batch of ``_BATCH``, the last one padded: OpenCV 5.0 crashed
        # (0xC0000409, 28 Sep 2026) when the batch size changed between calls.
        outs = {name: [] for name in self.outputs}
        for s in range(0, n, _BATCH):
            chunk = blob[s:s + _BATCH]
            if len(chunk) < _BATCH:
                chunk = np.concatenate([chunk, np.zeros((_BATCH - len(chunk),) + chunk.shape[1:], np.float32)])
            self.net.setInput(np.ascontiguousarray(chunk))
            for name, out in zip(self.outputs, self.net.forward(self.outputs)):
                outs[name].append(np.asarray(out).reshape(_BATCH, -1))
        got = {name: np.concatenate(v)[:n] for name, v in outs.items()}
        kind = _softmax(got["kind"])
        family = _softmax(got["family"])
        stripe = 1.0 / (1.0 + np.exp(-got["stripe"].reshape(n)))
        return kind, family, stripe


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


_LOADED: dict = {}


def load(path: Optional[Path] = None) -> Optional[BallNet]:
    """The shipped model, loaded once per process, or None if it is not there.

    ``BILLIARDS_BALLNET`` names another model file to use instead (to compare
    two), or ``off`` for none.
    """
    override = os.environ.get("BILLIARDS_BALLNET")
    if path is None and override:
        if override.lower() == "off":
            return None
        path = Path(override)
    path = Path(path) if path else MODEL_PATH
    key = str(path)
    if key not in _LOADED:
        try:
            _LOADED[key] = BallNet(path) if path.exists() else None
        except cv2.error:
            _LOADED[key] = None
    return _LOADED[key]
