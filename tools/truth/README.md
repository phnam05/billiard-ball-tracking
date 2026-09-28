# Answer keys for real footage

Real clips have no simulator behind them, so nothing said how well the tracker
did on them. It was judged by counting track ids, and on 28 Sep 2026 that
count compared two different things (DIARY, 28 Sep row 19). Each file here is
an answer key for one real clip, marked by eye, and
[`tools/real_eval.py`](../real_eval.py) scores a run against it.

The clips themselves are not committed (`.cache/`, `billiards-workspace/`);
`source` says where each one comes from.

## Format

```json
{
  "name": "us-open-3min",
  "what": "one line: what is in the clip and why it is here",
  "source": {"url": "https://www.youtube.com/watch?v=...", "part": ["11:48", "14:48"],
             "file": ".cache/truth/us-open-3min.mp4"},
  "settings": {"preset": "pool-9ft", "numbers": "1-9"},
  "frame_width": 1280,
  "match_px": 16,
  "segments": [
    {"from": 0, "to": 478, "kind": "play", "rack": 1},
    {"from": 479, "to": 505, "kind": "graphic"},
    {"from": 506, "to": 789, "kind": "replay"}
  ],
  "keyframes": [
    {"frame": 120, "balls": [["cue", 607.5, 498.9], [6, 534.0, 561.2], ["?", 700, 400]],
     "ignore": [[467, 411]]}
  ],
  "shots": [{"frame": 30, "potted": [], "note": "break"}],
  "marked_by": "who, how",
  "marked_on": "2026-09-28"
}
```

* **Frames** are numbered as `billiards.video.read_frames` numbers them from
  the start of the file, and **positions** are pixels in the frame as the
  app processes it (`frame_width` wide: 1280 for 720p). Ball centres, as seen.
* **`segments`** cover the whole clip, in order, without gaps:
  `play` (the whole table, or nearly, from any camera, live), `replay`,
  `closeup` (part of the table, a player), `crowd`, `graphic`
  (titles, wipes). `rack` numbers the racks among `play` segments; balls keep
  their identity within a rack.
* **`keyframes`**, about every 4 s of `play`, list **every ball that is at
  least half visible**: `"cue"`, its number, or `"?"` when the number cannot
  be told. Hidden balls are left out; so are keyframes where a rack is still
  tight or balls are a blur.
* **`ignore`** (optional): spots where a reported ball is neither right nor
  wrong -- a ball hidden behind a player (the tracker may rightly keep it,
  coasting), or held off the table by the referee.
* **`match_px`**: a reported ball within this many pixels of a marked one is
  that ball (about two thirds of a ball's diameter in the picture).
* **`shots`** (optional): the frame each shot is struck, and the balls potted.
