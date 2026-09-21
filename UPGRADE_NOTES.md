# Upgrade notes — v1 (2024) → v2

The original program worked, but only after hand-tuning HSV bounds for each ball
in each video, and it still lost the ball regularly. This rewrite removes the
tuning and replaces the "find the biggest coloured blob" approach with
calibration, detection, tracking and event analysis.

Everything below is measured on a synthetic break clip with exact ground truth
(`tools/make_synthetic_clip.py` renders it, `tools/evaluate.py` scores it), so
the numbers are reproducible rather than impressions.

---

## 1. Results

| Metric | v1 behaviour | v2 | Notes |
|---|---|---|---|
| Balls tracked | 2 (one cue + one object ball) | all of them (9 in the test clip, 16 supported) | |
| MOTA | n/a — could not be scored | **0.934** | standard MOT accuracy |
| Precision | 0.66 *(mid-rewrite measurement)* | **1.000** | zero false positives |
| Recall | 0.85 | **0.935** | |
| ID switches | identity was not even maintained | **1** over a full break | |
| Median position error | n/a | **0.157 in** | ball radius is 1.125 in |
| 95th-pct position error | n/a | 0.46 in | |
| Colour parameters to tune | 4 arrays, per ball, per video | **0** | cloth colour is measured |
| Pixel thresholds in the code | at least 6 | **0** | all thresholds are in inches |

The "v1 behaviour" column is indicative: v1 has no notion of multiple balls or
identity, so most MOT metrics are undefined for it.

---

## 2. Why v1 needed constant retuning

Five root causes, in rough order of how much pain each caused.

| # | Root cause | Consequence |
|---|---|---|
| 1 | **Detection was a hand-picked HSV box per ball.** `lower_bound = np.array([30, 20, 200]) - sense` etc. | Any change in lighting, camera, cloth or ball set invalidated the numbers. Red/pink balls could not be expressed at all, because hue wraps at 180 and a symmetric box clips half the distribution. |
| 2 | **No tracker.** Each frame independently took `max(contours, key=cv2.contourArea)`. | One missed detection broke the trajectory permanently. Two balls touching merged into one blob and the "ball" jumped to the merged centroid. Nothing survived an occlusion. |
| 3 | **Thresholds were in pixels.** `distance < 20` for collisions, `< 100` for the motion gate. | Meaningful only at one resolution, one zoom and one camera distance. On an angled view, no single value is even correct across one frame. |
| 4 | **Cloth colour came from three independent 1-D histograms.** `GetClothColor` took the argmax of the H, S and V histograms separately. | The joint mode of a colour distribution is not the product of its marginal modes, so the "cloth colour" was frequently a colour present nowhere in the image. |
| 5 | **Corners were the contour points nearest the image corners.** `Get_UL_Coord` and friends. | Pool table corners are cut away by the pockets, so the "corner" was wrong by the pocket radius; and the rule breaks entirely if the table is rotated or off-centre in frame. |

---

## 3. What replaced each part

### 3.1 Calibration is measured, not configured — `billiards/table.py`

* Cloth colour is estimated from **25 frames sampled across the whole clip**, not
  from frame 0. A player leaning over the rail spoils a minority of frames, and
  a median ignores them.
* The hue mode is found on a **smoothed circular histogram**, so red cloth
  (hue near 0/180) works.
* Window widths come from the **median absolute deviation**, which is immune to
  the ball pixels that inevitably leak into the sample. A plain standard
  deviation is not.
* Hue and saturation get tight windows, value gets a loose one, because hue and
  saturation barely move under lighting change and value moves a lot.
* The value window is **asymmetric** (wider downwards): a shadow only ever makes
  the cloth darker. Treating ball shadows as foreground is what welds
  neighbouring balls into a single blob.

### 3.2 Table geometry is fitted — `billiards/geometry.py`

* The four cushions are found as the **dominant edges** of the cloth contour,
  their supporting lines are **robustly re-fitted to all the contour points
  along them** (Huber, two rounds of outlier rejection), and the corners are the
  **intersections of those lines**.
  This recovers the true corner even though the pocket has eaten it. Measured on
  the synthetic clip: corner error dropped from **22 px → 1.4 px** when the line
  re-fit was added.
* Contours are extracted with `CHAIN_APPROX_NONE`, not `SIMPLE`. `SIMPLE`
  compresses a straight run of boundary pixels to its two endpoints, which left
  the line fit with almost no data.
* Corner ordering is by **angle around the centroid**, which is correct for a
  rotated table; the old "smallest x+y is top-left" rule is not.
* **Table orientation is detected.** v1 hard-coded `maxHeight = maxWidth * 2`,
  which silently transposed every clip where the table is wider than tall in
  frame — i.e. nearly every broadcast angle.
* The homography is computed **once**, and re-checked every 150 frames only to
  notice a camera cut. v1 recomputed it on every frame and never freed the
  buffers (see §4).

### 3.3 Ball detection is geometric, not chromatic — `billiards/detect.py`

The question changed from "which pixels are in this HSV box?" to **"what is on
the bed that is not cloth?"**. That removes colour tuning entirely. What remains
is decided by size and shape, and the **expected ball size is derived from the
homography at each image location** — so a ball at the far cushion is correctly
expected to be smaller than one at the near rail.

* **Pockets are excluded.** A pocket is a dark, round, ball-sized hole that never
  moves, so a "not cloth" detector sees it as a permanently stationary ball.
  This alone accounted for **4 phantom tracks on every frame** and dragged
  precision down to 0.66.
* **Clusters are split two ways**, because neither method covers both cases:
  * *distance-transform peaks* for balls with a sliver of cloth between them;
  * *fast radial symmetry voting* at the known ball radius for balls with no gap
    at all. A racked triangle is one solid mass whose distance transform peaks
    in the middle of the *triangle*; radial symmetry still finds the individual
    balls because each ball's circular edge votes for its own centre.
    On the test clip this took the rack from **0 of 8 balls found to 6 of 8**.
* **A blob thinner than a ball cannot contain one.** Gating on the maximum of
  the distance transform is what keeps the cue stick, the bridge hand and rail
  glare out, with no length or colour threshold to tune.
* Elongated, non-convex and non-circular blobs are rejected by shape.
* Ball identity uses a **CIE Lab colour signature** plus a white fraction
  (solid / stripe / cue / eight), sampled from the inner 62% of the disc to
  avoid the rim.

### 3.4 Tracking — `billiards/kalman.py`, `billiards/track.py`, `billiards/assignment.py`

* One **Kalman filter per ball, in table inches**. This matters: in the image, a
  ball rolling at constant speed appears to accelerate as it comes towards the
  camera, so a constant-velocity model fitted in pixel space is permanently
  wrong. After the homography it is right.
* The motion model includes **rolling friction** — velocity decays with a time
  constant — so a ball occluded for half a second is predicted to the right
  place instead of overshooting.
* **Adaptive process noise.** A fixed value cannot serve both requirements: low
  enough that a stationary ball does not accumulate phantom velocity, high
  enough to follow a collision. The filter measures how surprising each
  measurement is (normalised innovation squared) and loosens up **in the same
  frame** when something happens.
  Measured effect: reacting in-frame rather than one frame later improved MOTA
  0.929 → **0.934**, precision 0.997 → **1.000**, and cut worst-case position
  error 2.04 in → **1.83 in**, while letting the base noise stay low enough that
  a stationary ball reads 2.7 in/s instead of 9.8 in/s.
* **Globally optimal assignment** (Hungarian / Jonker–Volgenant) on a cost that
  combines predicted distance and colour distance. Greedy matching lets an early
  choice steal the detection a later track needed; colour is what keeps
  identities correct through a collision, where position alone is ambiguous.
  SciPy is used when present; an exact pure-NumPy implementation ships as a
  fallback (verified against SciPy on 400 random matrices, including
  rectangular and infeasible ones).
* **Track lifecycle**: tentative → confirmed → coasting → deleted, with a
  physical gate (`max_speed × dt + padding`). A track that disappears near a
  pocket is reported as **potted** rather than lost.

### 3.5 Events — `billiards/events.py`

`distance < 20` pixels became: closer than **1.12 ball diameters** *and* with a
positive closing speed. Requiring the balls to actually be approaching is what
stops two balls resting against each other from emitting a collision on every
frame. Cushion contacts, pots and shot starts are detected on the same physical
basis.

### 3.6 Output — `billiards/render.py`, `billiards/video.py`

* The overhead view is **drawn from table coordinates** rather than warped from
  camera pixels: crisp at any zoom, free to render, and it shows what the
  tracker actually believes.
* Coasted (predicted, unobserved) balls are drawn **dashed**, so belief is
  visually distinguishable from evidence.
* Per-frame track data streams to CSV; events and a run summary go to JSON.

---

## 4. Specific bugs fixed

| Bug | Where in v1 | Effect |
|---|---|---|
| Video writer created at the source frame size but fed 854×480 frames | `main.py` — `result = cv2.VideoWriter(..., size)` then `frame = cv2.resize(frame, (854,480))` | The saved file was unplayable. |
| The frame was written **before** anything was drawn on it | `result.write(frame)` sits above all the drawing code | Even a valid file would have contained no tracking overlay. |
| Output frame rate hard-coded to 10 | `cv2.VideoWriter(..., 10, size)` | Wrong playback speed; any timestamp derived from it was wrong. |
| `cv2.waitKey(0)` inside the main loop | `key = cv2.waitKey(0)` | Required a key press **per frame** — the tool could not play a video. |
| Input file name did not exist in the repo | `cv2.VideoCapture('kpc-break.mp4')` | The program as committed could not run. |
| Unbounded growth + O(n²) rescans | `contour_list.append(...)`, `areas.append(...)`, then `Indexer.get_index_of_max(areas)` every frame | Memory grew without limit and the loop slowed down as the clip went on. |
| Trajectory lists never bounded | `prev_center` / `oj_prev_center` | Same; also made the drawn trail grow forever. |
| `oj_past_center` used before assignment | compared against `None` in one branch, dereferenced in another | Crash on some inputs. |
| Table orientation assumed portrait | `maxHeight = (maxWidth * 2)` | Transposed table coordinates on landscape camera angles. |
| Corner detection defeated by pockets | `Get_UL_Coord` and friends | Corners wrong by the pocket radius. |
| `np.cross` on 2-D vectors | (introduced during the rewrite) | Removed in NumPy 2.0; replaced with an explicit scalar cross product. |

---

## 5. New capabilities

* **CLI** — `track`, `calibrate` and `dump-config` subcommands. Nothing needs a
  source edit. `python main.py clip.mp4 --show` just works.
* **Table presets** — 7/8/9 ft pool, snooker, carom. Carom tables correctly have
  no pockets.
* **`calibrate --save-preview`** — writes a PNG showing the detected table, the
  measured cloth mask and the resulting detections, so a bad calibration is
  visible in one glance instead of being guessed at.
* **`--table-corners`** — manual override for cases the automatic fit cannot
  handle (table partly out of frame, extreme angle).
* **CSV / JSON export** — per-frame positions in both inches and pixels, plus a
  full event log.
* **Ground-truth simulator and MOT scorer** — `tools/make_synthetic_clip.py`
  and `tools/evaluate.py`. Any future change can be measured rather than
  eyeballed.
* **33 tests**, including end-to-end accuracy assertions.
* Works **headless** (no display required).

---

## 6. Reproducing the numbers

```bash
pip install -r requirements.txt

python tools/make_synthetic_clip.py \
    --out out/break.mp4 --ground-truth out/gt.csv

python main.py track out/break.mp4 --csv out/tracks.csv --json out/run.json

python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv
```

And the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q            # everything
pytest -q -m "not slow"   # unit tests only
```

---

## 7. If you do need to tune something

You should not need to touch colour at all. The knobs most likely to matter, in
order:

| Setting | When to change it |
|---|---|
| `table.preset` | Your table is not a 9 ft pool table. |
| `table.bed_margin_ball_diameters` | Balls right against the cushion are being missed (lower it), or the rail is generating blobs (raise it). |
| `detector.pocket_exclusion_ball_diameters` | Balls near a pocket are being missed (lower it), or pocket jaws still detect (raise it). |
| `cloth.hue_sigmas` / `sat_sigmas` | The cloth mask is visibly wrong in `calibrate --save-preview`. |
| `tracker.max_speed_in_s` | Only if you film something faster than a pool break. |

Start with `python main.py calibrate <video> --save-preview calib.png`. Almost
every tracking problem is visible there first.

---

## 8. Known limitations

* A **tightly racked triangle** yields roughly 6 of 8 balls while it is static.
  All of them are picked up as soon as the rack separates. Balls in a rack are
  genuinely ambiguous — adjacent balls of similar colour share no visible edge.
* A ball **resting in the pocket jaws** is inside the excluded region and is not
  detected. It is reported as `potted`.
* **Cloth-coloured balls** (a green ball on green cloth) are hard by
  construction; the detector is looking for "not cloth".
* The automatic fit needs the **whole bed visible**. Use `--table-corners` if it
  is not.
* Motion blur at high shutter speeds smears a fast ball into an ellipse; the
  aspect-ratio gate can reject it, and the track coasts through instead.

---

## 9. File map

```
billiards/
  config.py      physical-unit configuration, presets, YAML/JSON
  geometry.py    quad fitting, homography, perspective-aware scale
  table.py       cloth colour measurement, table calibration
  detect.py      ball detection, cluster splitting, colour signatures
  kalman.py      constant-velocity + friction filter, adaptive noise
  assignment.py  Hungarian assignment (+ pure-NumPy fallback)
  track.py       track lifecycle and data association
  events.py      collisions, cushions, pots, shot starts
  render.py      annotated view and synthetic overhead diagram
  video.py       input, output, CSV/JSON export
  pipeline.py    orchestration and the whole-video driver
  cli.py         command line interface
tools/
  make_synthetic_clip.py   physics simulator + renderer + ground truth
  evaluate.py              MOT scoring
legacy/          the original v1 code, kept for comparison
tests/           33 tests
```
