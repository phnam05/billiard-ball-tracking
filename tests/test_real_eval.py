"""The real-footage scorer (tools/real_eval.py), on a hand-made answer key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import real_eval  # noqa: E402


def _row(frame, tid, x, y, label="", number=""):
    return {"frame": str(frame), "track_id": str(tid), "x_px": str(x), "y_px": str(y),
            "label": label or f"#{tid}", "number": number}


TRUTH = {
    "name": "toy",
    "match_px": 10,
    "segments": [
        {"from": 0, "to": 9, "kind": "play", "rack": 1},
        {"from": 10, "to": 14, "kind": "replay"},
        {"from": 15, "to": 29, "kind": "play", "rack": 1},
    ],
    "keyframes": [
        {"frame": 5, "balls": [["cue", 100, 100], [3, 200, 100], ["?", 300, 100]],
         "ignore": [[400, 100]]},
        {"frame": 20, "balls": [["cue", 110, 100], [3, 200, 100]]},
    ],
}


def _rows():
    rows = {}
    for f in range(0, 30):
        if 10 <= f <= 11:
            rows[f] = [_row(f, 1, 100, 100, "CUE")]   # drawn during the replay
        elif 12 <= f <= 14:
            continue                                   # ...and then not
        elif f == 5:
            rows[f] = [
                _row(f, 1, 103, 101, "CUE"),          # the cue ball, named
                _row(f, 2, 199, 98, "7", "7"),        # the 3, named wrongly
                _row(f, 3, 301, 100),                 # the unknown ball
                _row(f, 4, 401, 100),                 # on the ignored spot
                _row(f, 5, 600, 300),                 # a phantom
            ]
        elif f == 20:
            rows[f] = [
                _row(f, 1, 111, 99, "CUE"),
                _row(f, 6, 200, 101, "3", "3"),       # the 3 again, under a new id
            ]
        elif f < 25:
            rows[f] = [_row(f, 1, 100, 100, "CUE")]
    return rows                                        # frames 25-29: nothing drawn


def test_score_counts_found_phantoms_names_and_ids():
    r = real_eval.score(TRUTH, _rows())
    assert r["balls"] == 5
    assert r["found"] == 1.0
    assert r["phantoms_per_keyframe"] == 0.5          # one phantom; the ignored spot is not one
    assert r["real"] == pytest.approx(5 / 6, abs=1e-3)
    # Known balls found: cue twice, the 3 twice; the "?" does not count.
    assert r["named_right"] == 0.75 and r["named_wrong"] == 0.25
    assert r["ids_per_ball"] == 1.5                   # the cue kept id 1, the 3 had 2 and 6
    assert r["swaps"] == 0
    assert r["score"] == pytest.approx(1 - (0 + 1 + 1) / 5, abs=1e-3)
    assert r["off_play"] == {"replay": 0.4}
    assert r["paused"] == pytest.approx(5 / 25, abs=1e-3)


def test_a_ball_out_of_reach_is_missed_not_found():
    rows = {5: [_row(5, 1, 100, 115, "CUE")]}         # 15 px away, gate 10
    r = real_eval.score({**TRUTH, "keyframes": TRUTH["keyframes"][:1]}, rows)
    assert r["found"] == 0.0
    assert r["phantoms_per_keyframe"] == 1.0


def test_one_id_on_two_balls_is_a_swap():
    rows = {
        5: [_row(5, 1, 100, 100), _row(5, 2, 200, 100)],
        20: [_row(20, 2, 110, 100), _row(20, 1, 200, 100)],
    }
    r = real_eval.score(TRUTH, rows)
    assert r["swaps"] == 2
    assert r["ids_per_ball"] == 2.0


def test_identity_reads_the_label_and_number():
    assert real_eval.identity({"label": "CUE", "number": ""}) == "cue"
    assert real_eval.identity({"label": "8", "number": ""}) == 8
    assert real_eval.identity({"label": "12", "number": "12"}) == 12
    assert real_eval.identity({"label": "#4", "number": ""}) is None


def test_every_answer_key_is_well_formed():
    """Segments cover each clip without gaps, keyframes sit in play."""
    keys = real_eval.all_truths()
    assert keys, "no answer keys in tools/truth/"
    for t in keys:
        segs = t["segments"]
        assert segs[0]["from"] == 0
        for a, b in zip(segs, segs[1:]):
            assert b["from"] == a["to"] + 1, (t["name"], a, b)
        for kf in t["keyframes"]:
            seg = next(s for s in segs if s["from"] <= kf["frame"] <= s["to"])
            assert seg["kind"] == "play", (t["name"], kf["frame"])
            for ball in kf["balls"]:
                assert ball[0] == "cue" or ball[0] == "?" or 1 <= int(ball[0]) <= 15, (t["name"], ball)
