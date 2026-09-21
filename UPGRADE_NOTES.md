# Upgrade notes — v1 (2024) → v2

The original program worked, but only after hand-tuning HSV bounds for each ball
in each video, and it still lost the ball regularly. This rewrite removes the
tuning and replaces "find the biggest coloured blob" with calibration,
detection, tracking and event analysis.

Two things make the numbers below trustworthy rather than impressions:

* `tools/make_synthetic_clip.py` renders a physically simulated break from a
  virtual camera **with exact ground truth**, and `tools/evaluate.py` scores a
  run against it with the standard MOT protocol;
* everything was then run against the three real match clips already in this
  repo, which is where most of the interesting bugs turned up.

---

## 1. Results

| Metric | v1 | v2 |
|---|---|---|
| Balls tracked | 2 (one cue + one object ball) | all of them |
| MOTA | not measurable | **0.936** |
| Precision / recall | — | **1.000** / 0.937 |
| ID switches over a full break | identity was not maintained | **2** |
| Median position error | — | **0.150 in** (ball radius 1.125 in) |
| 95th-pct position error | — | 0.40 in |
| Colour parameters to tune | 4 arrays, per ball, per video | **0** |
| Pixel thresholds in the code | at least 6 | **0** |
| Speed (1280×720) | n/a (blocked on a key press per frame) | **23 fps** |
| Handles a camera cut | no | yes — pauses, re-finds the table |

"v1" has no notion of multiple balls or identity, so most MOT metrics are
undefined for it.

---

## 2. Why v1 needed constant retuning

| # | Root cause | Consequence |
|---|---|---|
| 1 | **Detection was a hand-picked HSV box per ball.** `lower_bound = np.array([30, 20, 200]) - sense` | Any change in lighting, camera, cloth or ball set invalidated the numbers. Red/pink balls were inexpressible: hue wraps at 180, so a symmetric box clips half the distribution. |
| 2 | **No tracker.** Each frame independently took `max(contours, key=cv2.contourArea)`. | One missed detection broke the trajectory permanently. Two touching balls merged into one blob and the "ball" jumped to the merged centroid. Nothing survived an occlusion. |
| 3 | **Thresholds were in pixels.** `distance < 20` for collisions, `< 100` for the motion gate. | Meaningful at exactly one resolution, zoom and camera distance. On an angled view, no single value is correct even across one frame. |
| 4 | **Cloth colour came from three independent 1-D histograms.** | The joint mode of a colour distribution is not the product of its marginal modes, so the "cloth colour" was often a colour present nowhere in the image. |
| 5 | **Corners were the contour points nearest the image corners.** | Pool table corners are cut away by the pockets, so the "corner" was wrong by the pocket radius — and the rule collapses entirely if the table is rotated in frame. |

---

## 3. What replaced each part

### 3.1 Calibration is measured, not configured — `billiards/table.py`

* Cloth colour is estimated from **25 frames sampled across the whole clip**.
  A player leaning over the rail spoils a minority of frames; a median ignores
  them.
* The hue mode is found on a **smoothed circular histogram**, so red cloth
  (hue near 0/180) works.
* Window widths come from the **median absolute deviation**, immune to the ball
  pixels that inevitably leak into the sample.
* Hue and saturation get tight windows, value a loose one, because hue and
  saturation barely move under lighting change and value moves a lot.
* The value window is **asymmetric** — wider downwards, because a shadow only
  ever makes cloth darker — but with a **multiplicative floor**. A shadow is a
  roughly 50% darkening; anything darker is an object. Without that floor the
  window swallowed dark balls whole, and on grey cloth the black ball simply
  never appeared.
* The estimate is then **refined inside the region it selected**. Measuring over
  the whole frame lets a background that shares the cloth's hue — blue banners
  behind a blue table, which is most tournament footage — widen the robust
  spread until the window accepts everything.

### 3.2 Table geometry is fitted — `billiards/geometry.py`

* The four cushions are found as the **dominant edges** of the cloth contour,
  their supporting lines **robustly re-fitted to every contour point along
  them** (Huber, two rounds of outlier rejection), and the corners taken as the
  **intersections of those lines**. This recovers a corner the pocket has eaten.
  Measured: corner error **22 px → 1.4 px**.
* Contours use `CHAIN_APPROX_NONE`, not `SIMPLE`. `SIMPLE` compresses a straight
  run of boundary pixels to its two endpoints, leaving the line fit with almost
  no data.
* Corner ordering is by **angle around the centroid**, correct for a rotated
  table; "smallest x+y is top-left" is not.
* **Orientation is decided by projective geometry, not pixel counts.** This one
  matters: filmed from behind an end rail — the standard pool camera angle — the
  100-inch length is foreshortened into *fewer* pixels than the 50-inch near
  cushion, so "the longer edge is the long side" picks the wrong axis and every
  distance is out by a factor of two. Instead both assignments are tried and the
  one a real camera could have produced is kept: a homography onto a rectangle
  of the right shape yields a positive, self-consistent focal length.
  (Only the equal-column-norm constraint discriminates; the orthogonality one is
  invariant under exactly the swap that distinguishes the two orientations.)

### 3.3 Ball detection is geometric, not chromatic — `billiards/detect.py`

The question changed from "which pixels are in this HSV box?" to **"what is on
the bed that is not cloth?"**, which removes colour tuning entirely. What
remains is decided by size and shape, with the **expected ball size derived from
the homography at each image location**.

* **A ball is a sphere, not a disc painted on the cloth.** A flat disc viewed at
  a grazing angle is squashed into a thin ellipse; a sphere is not squashed at
  all, because its silhouette is always a circle. Using the flat-plane scale
  under-predicted ball size by 40–80% on real footage, so every ball looked like
  a ~3× cluster and got split into two phantom halves. Balls now use the largest
  singular value of the local Jacobian; flat features (pockets, markings) keep
  the area-preserving scale, and the pocket exclusion zone is projected as the
  ellipse it actually is.
* **Pockets are excluded.** A pocket is a dark, round, ball-sized hole that
  never moves, so a "not cloth" detector sees it as a permanently stationary
  ball. This alone was **4 phantom tracks on every frame**, and dragged
  precision to 0.66.
* **Clusters are split two ways**, because neither method covers both cases:
  * *distance-transform peaks* for balls with a sliver of cloth between them;
  * *radial-symmetry voting* at the known ball radius for balls with no gap at
    all. A racked triangle is one solid mass whose distance transform peaks in
    the middle of the *triangle*, nowhere near any ball.
  The voting runs on the **Di Zenzo colour gradient**, not the grayscale one:
  two touching balls of different colours at similar lightness have almost no
  brightness edge, and an edge the splitter cannot see is a ball it cannot find.
  That change alone took recall 0.898 → **0.953** and ID switches 4 → 1.
* **A blob thinner than a ball cannot contain one.** Gating on the maximum of
  the distance transform is what keeps the cue stick, the bridge hand and rail
  glare out — with no length or colour threshold to tune.
* Ball identity uses a **CIE Lab colour signature** plus a white fraction and a
  high percentile of chroma. The last one separates the cue ball from a striped
  ball: both are mostly white, but a stripe carries one strongly coloured band
  and the cue ball carries none. Without it, every stripe is labelled "CUE".

### 3.4 Tracking — `billiards/kalman.py`, `track.py`, `assignment.py`

* One **Kalman filter per ball, in table inches**. In the image, a ball rolling
  at constant speed appears to accelerate as it approaches the camera, so a
  constant-velocity model fitted in pixel space is permanently wrong. After the
  homography it is right.
* The motion model includes **rolling friction**, so a ball occluded for half a
  second is predicted to the right place instead of overshooting.
* **Adaptive process noise.** A fixed value cannot be both low enough that a
  stationary ball does not accumulate phantom velocity and high enough to follow
  a collision. The filter measures how surprising each measurement is
  (normalised innovation squared) and loosens **within the same frame**.
  Measured: reacting in-frame rather than one frame later took MOTA 0.929 →
  0.934 and precision 0.997 → 1.000, while letting the base noise stay low
  enough that a stationary ball reads 2.7 in/s instead of 9.8 in/s.
* **Globally optimal assignment** (Hungarian / Jonker–Volgenant) on predicted
  distance combined with colour distance. Greedy matching lets an early choice
  steal the detection a later track needed; colour is what keeps identities
  correct through a collision. SciPy is used when present; an exact pure-NumPy
  implementation ships as a fallback, verified against SciPy on 400 random
  matrices including rectangular and infeasible ones.
* **Track lifecycle**: tentative → confirmed → coasting → deleted, with a
  physical gate. A track that disappears near a pocket is reported as **potted**
  rather than lost.

### 3.5 Camera cuts — `billiards/pipeline.py`

Broadcast pool cuts constantly, to replays, overhead angles and player
close-ups. With a stale homography still applied, a cut to a crowd shot produced
dozens of "balls" sitting on spectators with trajectories drawn between them.

The signal used is one the detector computes anyway: **how much of the bed
polygon is still cloth-coloured**. On the right shot it sits at 0.95–0.98; on
the test clip it collapses to 0.38 across the cut. Below a fraction of its
calibrated value, tracking is suspended — nothing at all is the correct output
for a frame with no table in it — and the table is then re-found from **several
agreeing frames**, as the initial calibration does, rather than from one frame
of a crossfade. A candidate table is adopted only if its polygon is mostly
cloth, which is what separates a table from a sponsor banner.

### 3.6 Events — `billiards/events.py`

`distance < 20` pixels became: closer than **1.12 ball diameters** *and* with a
positive closing speed. Requiring the balls to be approaching is what stops two
balls resting against each other from emitting a collision every frame. Cushion
contacts, pots and shot starts are detected on the same physical basis.

### 3.7 Output — `billiards/render.py`, `video.py`

* The overhead view is **drawn from table coordinates** rather than warped from
  camera pixels: crisp at any zoom, free to render, and it shows what the
  tracker actually believes.
* Coasted (predicted, unobserved) balls are drawn **dashed**, so belief is
  visually distinguishable from evidence.
* Per-frame track data streams to CSV; events and a run summary go to JSON.

---

## 4. Specific bugs fixed

| Bug | Where | Effect |
|---|---|---|
| Video writer created at the source frame size but fed 854×480 frames | `main.py` | The saved file was unplayable. |
| The frame was written **before** anything was drawn on it | `result.write(frame)` above all drawing | Even a valid file would have contained no overlay. |
| Output frame rate hard-coded to 10 | `cv2.VideoWriter(..., 10, size)` | Wrong playback speed; every derived timestamp wrong. |
| `cv2.waitKey(0)` inside the main loop | | Required a key press **per frame** — it could not play a video. |
| Input file name did not exist in the repo | `VideoCapture('kpc-break.mp4')` | The program as committed could not run. |
| Unbounded growth + O(n²) rescans | `contour_list.append(...)` then `get_index_of_max(areas)` every frame | Memory grew without limit and the loop slowed as the clip went on. |
| `oj_past_center` used before assignment | | Crash on some inputs. |
| Table orientation assumed portrait | `maxHeight = maxWidth * 2` | Transposed table coordinates on landscape angles. |
| Corner detection defeated by pockets | `Get_UL_Coord` and friends | Corners wrong by the pocket radius. |
| Crash when a frame has no detections | *(introduced in the rewrite)* | `np.array([])` is shape `(0,)`, not `(0,2)`; found by the camera-cut test. |
| `np.cross` on 2-D vectors | *(introduced in the rewrite)* | Removed in NumPy 2.0. |
| Run totals reset on recalibration | *(introduced in the rewrite)* | The summary described only the stretch since the last cut. |

---

## 5. New capabilities

* **CLI** — `track`, `calibrate`, `dump-config`. Nothing needs a source edit.
* **Table presets** — 7/8/9 ft pool, snooker, carom (carom correctly has no
  pockets).
* **`calibrate --save-preview`** — a PNG showing the detected table, the measured
  cloth mask and the resulting detections, so a bad calibration is visible at a
  glance instead of guessed at.
* **`--table-corners`** — manual override for cases the automatic fit cannot
  handle.
* **CSV / JSON export** — per-frame positions in inches *and* pixels, plus a
  full event log.
* **Ground-truth simulator and MOT scorer**, so any future change is measured
  rather than eyeballed.
* **38 tests**, including end-to-end accuracy assertions and a camera-cut test.
* Works **headless**.

---

## 6. Performance

2.4× faster than the first working version of the rewrite, with output identical
to every printed digit:

* the local magnification is tabulated once on a 65×33 grid in table
  coordinates and interpolated, instead of two perspective transforms and an SVD
  per detection per frame;
* colour signatures sample 64 points spread over the disc rather than every
  pixel inside it, making their cost independent of apparent ball size;
* the association cost matrix is built in one vectorised pass instead of a
  Python double loop.

9.8 → **23.3 fps** on 1280×720 broadcast footage.

---

## 7. Reproducing the numbers

```bash
pip install -r requirements.txt

python tools/make_synthetic_clip.py --out out/break.mp4 --ground-truth out/gt.csv
python main.py track out/break.mp4 --csv out/tracks.csv --json out/run.json
python tools/evaluate.py --gt out/gt.csv --tracks out/tracks.csv

pip install -r requirements-dev.txt
pytest -q                  # all
pytest -q -m "not slow"    # unit tests only
```

---

## 8. If you do need to tune something

You should not need to touch colour at all. In rough order of likelihood:

| Setting | When to change it |
|---|---|
| `table.preset` | Your table is not a 9 ft pool table. |
| `table.bed_margin_ball_diameters` | Balls against the cushion are missed (lower), or the rail generates blobs (raise). |
| `detector.pocket_exclusion_ball_diameters` | Balls near a pocket are missed (lower), or pocket jaws still detect (raise). |
| `cloth.hue_sigmas` / `sat_sigmas` | The cloth mask looks visibly wrong in `calibrate --save-preview`. |
| `table.view_change_coverage_ratio` | Tracking pauses too eagerly on heavy occlusion (lower), or keeps going through cuts (raise). |
| `tracker.max_speed_in_s` | Only if you film something faster than a pool break. |

Start with `python main.py calibrate <video> --save-preview calib.png`. Almost
every tracking problem is visible there first.

---

## 9. Known limitations

* A **tightly racked triangle** yields roughly 6 of 8 balls while it is static;
  all of them appear as soon as the rack separates. Adjacent balls of similar
  colour genuinely share no visible edge.
* A ball **resting in the pocket jaws** is inside the excluded region and is
  reported as `potted`.
* **Cloth-coloured balls** are hard by construction — the detector looks for
  "not cloth".
* The automatic fit needs the **whole bed visible**; use `--table-corners`
  otherwise.
* Motion blur at high shutter speeds smears a fast ball into an ellipse, which
  the aspect gate can reject; the track coasts through instead.
* A **cue stick lying across the bed with a hand on it** can occasionally form a
  ball-sized blob and start a short-lived track.

---

## 10. File map

```
billiards/
  config.py      physical-unit configuration, presets, YAML/JSON
  geometry.py    quad fitting, homography, orientation, perspective scale
  table.py       cloth colour measurement, table calibration
  detect.py      ball detection, cluster splitting, colour signatures
  kalman.py      constant-velocity + friction filter, adaptive noise
  assignment.py  Hungarian assignment (+ pure-NumPy fallback)
  track.py       track lifecycle and data association
  events.py      collisions, cushions, pots, shot starts
  render.py      annotated view and synthetic overhead diagram
  video.py       input, output, CSV/JSON export
  pipeline.py    orchestration, camera-cut handling, whole-video driver
  cli.py         command line interface
tools/
  make_synthetic_clip.py   physics simulator + renderer + ground truth
  evaluate.py              MOT scoring
legacy/          the original v1 code, kept for comparison
tests/           38 tests
```
