# Billiard Ball Tracking

Tracks every ball on a pool table through a video — trajectories, collisions,
cushion contacts and pots — **without tuning colour thresholds for each clip**.

The cloth colour is measured from the video, the table is found and rectified,
and from then on every threshold in the pipeline is expressed in inches rather
than pixels. That is what makes it work on a new clip without being re-tuned.

![tracking demo](docs/images/demo_synthetic.png)

*Nine balls tracked through a break: per-ball trajectories, a contact point, and
a top-down diagram drawn from the tracker's own table coordinates. The clip is
the built-in physics simulator, so every position here is checkable against
ground truth.*

> Originally a final project for ENG 301 – Computer Vision at Fulbright
> University Vietnam, Spring 2024. Rewritten since; see
> [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md) for what changed and why, and
> [`legacy/`](legacy/) for the original code.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+. Only NumPy and OpenCV are required; SciPy and PyYAML are optional
(an exact pure-NumPy assignment solver and JSON config ship as fallbacks).

## Use

```bash
# Watch it work
python main.py clip.mp4 --show

# Save an annotated video plus data
python main.py clip.mp4 -o out.mp4 --csv tracks.csv --json run.json

# Check the calibration before a long run — this answers most questions
python main.py calibrate clip.mp4 --save-preview calib.png

# A different table
python main.py clip.mp4 --preset snooker-12ft
```

While a window is open: `space` pauses, `c` clears the trails, `q` quits.

### Outputs

* **Annotated video** — balls circled in their own colour with stable IDs,
  trajectories, contact points, and a top-down diagram inset. A ball the tracker
  is *predicting* rather than *seeing* is drawn dashed.
* **`--csv`** — one row per ball per frame: position in table inches and in
  image pixels, velocity, speed, track state.
* **`--json`** — calibration report, run statistics, the full event log
  (collisions with closing speed, cushion contacts, pots, balls struck) and a
  **shot log**: the raw events grouped into shots, the way a player reads them.

```
[billiards] 2 shot(s):
[billiards]   shot 1: CUE struck -- hit #4 first -- nothing potted -- 1.9s
[billiards]   shot 2: #2 struck -- potted #3, #20, #14, #18 -- 5.4s
```

A shot opens when a ball goes from rest to struck and closes once every ball has
settled — the same definition a referee uses — so it needs no timer or threshold
of its own.

### Options worth knowing

| Flag | Meaning |
|---|---|
| `--preset` | `pool-9ft` (default), `pool-8ft`, `pool-7ft`, `snooker-12ft`, `carom-10ft` |
| `--start` / `--end` | Process a time range, in seconds |
| `--table-corners x1,y1,...` | Override the automatic table fit |
| `--max-width` | Downscale for speed; accuracy is scale-invariant |
| `--debug` | Also show the detection mask |
| `--no-overhead` | Hide the top-down inset |

Full configuration: `python main.py dump-config config.yaml`, edit, then
`--config config.yaml`. Every value is in inches, seconds or a ratio — never
pixels.

## How it works

```
frame ──► cloth mask ──► "on the bed but not cloth" ──► size/shape gates ──► detections
              ▲                                              ▲
     measured from 25 frames                    expected ball size, from the
     sampled across the clip                    homography, at that image point

detections ──► Hungarian assignment (distance + colour) ──► Kalman filter per ball
                                                            (table inches, with
                                                             rolling friction)
                    │
                    └──► collisions · cushions · pots · shot starts
```

1. **Calibrate.** Measure the cloth colour robustly across the clip, then
   re-measure inside the region that selects (otherwise a background sharing the
   cloth's hue widens the window until it accepts everything). Fit the four
   cushions as lines and intersect them for the corners, since pockets eat the
   actual corners. Decide which way round the table is by which assignment a
   real camera could have produced — filmed down its length, a pool table's
   100-inch side covers *fewer* pixels than its 50-inch one.
2. **Detect.** Anything on the bed that is not cloth. Pockets are excluded
   geometrically. Ball size comes from the homography and is computed for a
   **sphere**, which is not foreshortened the way a painted disc is. Touching
   balls are separated by distance-transform peaks plus radial-symmetry voting
   on the colour gradient, which is what lets a racked cluster be resolved.
3. **Track.** One Kalman filter per ball in table coordinates, with rolling
   friction and process noise that loosens the moment something unexpected
   happens; globally optimal assignment on position *and* colour; coasting
   through occlusions instead of dying.
4. **Analyse.** Events in physical units: contact within 1.12 ball diameters
   *with a positive closing speed*, cushion bounces, pots — then grouped into
   shots by ball motion.
5. **Survive the cut.** Broadcasts change angle mid-clip. When the bed stops
   looking like cloth, tracking pauses rather than reporting balls in the crowd,
   and the table is re-found from several agreeing frames.

## Accuracy

Measured against a physically simulated clip with exact ground truth:

| | |
|---|---|
| MOTA | **0.936** |
| Precision / recall | **1.000** / 0.937 |
| ID switches over a full break | **2** |
| Median position error | **0.150 in** (ball radius is 1.125 in) |
| Speed, 1280x720 | **23 fps** |

Reproduce it:

```bash
python tools/make_synthetic_clip.py --out out/break.mp4 --ground-truth out/gt.csv
python main.py track out/break.mp4 --csv out/tracks.csv
python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv
```

## Library use

```python
from billiards import Config, RunOptions, run

summary = run(Config(), RunOptions(video="clip.mp4", output="out.mp4"))
print(summary["events"])
```

Or drive it frame by frame with `billiards.pipeline.build_pipeline` and
`TrackingPipeline.process`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                  # all 44
pytest -q -m "not slow"    # unit tests only
```

## Limitations

Racked balls resolve to roughly 6 of 8 while the rack is static (adjacent balls
of similar colour share no visible edge); a ball sitting in the pocket jaws is
reported as potted; a ball the same colour as the cloth is hard by construction.
See [`UPGRADE_NOTES.md §9`](UPGRADE_NOTES.md#9-known-limitations).
