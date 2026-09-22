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
  repo, which is where most of the interesting bugs turned up. Those clips have
  no ground truth, so how *noisy* their output is — phantom tracks, speed
  estimates that swing between frames, events for things that did not happen —
  is scored separately by `tools/run_report.py`, against physics rather than
  against a reference, and every run of it is kept in `reports/run-log.json`.

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
| Speed (1280×720) | n/a (blocked on a key press per frame) | **23 fps**, 27 on rewrapped broadcast footage |
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
  the distance transform is what keeps the cue stick and rail glare out — with
  no length or colour threshold to tune.
* **A split blob has to look like a group of balls, and a bridge hand manages
  it neither way.** A hand defeats every other test in the list: its knuckles
  are ball-thick, so the distance-transform gate passes the blob, and the
  splitter finds three convincing round peaks inside it. Two things separate
  the cases, and a blob only has to manage one.

  *It accounts for itself* — a group of touching balls is a union of discs of
  one known radius, so once the discs are drawn there should be nothing
  ball-thick left over. What *is* left over may be thin: a cue shaft, the cast
  shadow welding two balls together, a sleeve. None of those could hide a ball.

  *Or its discs sit on real ball edges.* Failing the first test does not prove
  the blob is not balls — it may be balls the splitter could not separate. A
  racked triangle of same-coloured neighbours yields five of eight, so its
  discs account for 0.60 of it, and rejecting on that alone **cost nine points
  of recall** against ground truth (0.937 → 0.844) before the second test was
  added. Those five still sit on unmistakable circular edges, which knuckles do
  not. Measured over four clips, the median rim contrast of a blob's candidates
  is **64–115 for an under-split rack and 62–65 for a well-split pair, against
  16–46 for a hand**; the threshold sits in that gap.
* **A ball ends at its rim.** One radius out there is cloth, or another ball,
  but never more of the same ball, so the colour step across the rim is large
  in almost every direction — 23–63 Lab units for real balls across the three
  clips. A disc drawn inside a hand, a forearm or a sleeve scores 5–19, because
  the material simply continues. Taking the median over 24 directions is what
  keeps a ball that is half hidden behind another, or clipped by the bed edge,
  from failing it.
* **Which ball is the cue ball is a question about the whole set**, not about
  each ball alone: a table has exactly one cue ball and one 8. Classifying
  independently produced two cue balls and four 8 balls on a real clip, because
  grey cloth pushes several balls into "dark and colourless" at once. The roles
  are now assigned across all confirmed tracks at once, with enough stickiness
  that two similar balls do not trade the label back and forth.
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
* **Coasting is extrapolation, and has to be paid for.** How far a track may be
  extrapolated is capped by how much evidence built its motion model: one frame
  of coasting per frame actually observed. A ball watched for five hundred
  frames still gets the full window, which it needs, because the player's body
  hides it for most of a stroke. A blob that looked like a ball three frames
  running gets three — where before it drew forty-five frames of confident
  trajectory on the strength of nothing. On the real clips the tracks this
  removes had six observations and sixty frames of invented path.

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

Coverage alone is a lagging signal, though, and a **dissolve** is what exposes
it: a broadcast crossfades between two shots of the same sport, so the incoming
angle is mostly cloth too and the outgoing bed polygon stays cloth-coloured
well into the transition. On `fedor_shot.mp4` coverage decays from 0.98 to 0.38
over nineteen frames, and tracking ran for twelve of them with the crowd, the
rails and a second table all inside a stale bed polygon. Those twelve frames
created **thirty phantom tracks** — half the clip's total.

So the bed is also watched for **wholesale change frame to frame**, which is a
different question: not "does this still look like cloth?" but "is this still
the same picture?". The two are complementary because of how little of a table
a game actually moves. Measured across the three clips, the busiest frame of
play repaints 6% of the bed and a typical one under 2%; a cut or a dissolve
repaints **16–32%** in a single frame. That is acted on immediately, with no
patience — a cut is not ambiguous — and recovery does not even start until the
picture settles, because a quad fitted from a frame that is half one shot and
half another describes neither.

Coming back is now free when the view comes back **unchanged** — a replay, a
dissolve that resolves to the shot it started from, a hand over the lens. If
the recovered polygon is the table we were already calibrated to, tracking
resumes on the existing tracks instead of rebuilding them, so ball identities
survive the interruption rather than being renumbered on the other side of it.

### 3.5a A repeated frame is not a measurement — `billiards/pipeline.py`

All three sample clips are **25 fps content rewrapped at 37.7 fps**, so one
frame in three is a copy of the one before it. Nothing in the file says so, and
the pipeline used to measure each copy as if it were new evidence.

That is double counting, and it is worse than it sounds. A copy tells the
filter the ball did not move over 1/37.7 s, which collapses its velocity
estimate; the next real frame then hands it one and a half frames of travel at
once. On `fedor_shot.mp4` a cue ball rolling smoothly at about 100 in/s was
reported at 30, 75, 9, 124 and 160 in/s on successive frames. The wreckage
downstream:

* the estimate dropped below the at-rest threshold in the middle of the roll,
  so a **second "struck" event** fired for a ball that had never stopped;
* the ball was stepped over the contact window, so the shot **potted a ball and
  reported no collision**;
* blobs on the player's gloved hand reached the three hits that confirm a
  track, because three copies of one picture counted as three sightings — which
  is where `#9` and `#10` came from.

The test is the frame-difference the cut detector already computes, read at the
other end of the scale: not "was the bed repainted?" but "did the bed change at
all?". A copy moves **no** bed pixel by more than an eighth of full scale,
measured across the colour channels — zero, on all three clips — while one ball
shifting by a single pixel repaints fifty. So a frame that moved less than two
hundredths of a ball area is replayed rather than measured, and the next frame
that *is* measured gets the whole interval as its `dt`. The geometry was never
wrong; only the clock was.

Colour rather than luminance, because a ball can differ from the cloth in hue
and barely at all in brightness, and that ball moving is precisely what must
not be mistaken for nothing happening.

At most four frames in a row are dropped. The bed is genuinely still between
shots — two seconds of it on these clips — and extrapolating a Kalman filter
across that in one step makes its association gate wider than the table.

Replayed frames still produce their CSV row and their frame of annotated video,
so a copy in the input is a copy in the output; what they do not produce is a
second measurement or a second event.

### 3.6 Events and shots — `billiards/events.py`, `billiards/shots.py`

`distance < 20` pixels became: closer than **1.25 ball diameters** *and* with a
gap that is actually shrinking. Requiring the balls to be approaching is what
stops two balls resting against each other from emitting a collision every
frame — those are as close as it gets, forever. Cushion contacts, pots and
balls struck are detected on the same physical basis.

Contact is measured **over the frame, not at the end of it**. Testing only
where the balls are now cannot work at the speeds a break reaches: two balls
are in contact across a shell 0.27 in thick, from 1.12 diameters apart down to
touching, and a cue ball crossing the table covers three or four inches between
frames, so it lands inside that shell about one time in fifteen. On
`fedor_shot.mp4` the gap between the cue ball and the ball it pocketed read
3.72 in on one frame and 2.53 in on the next, against a 2.52 in threshold. Both
balls travel in a straight line over one frame, though, so their closest
approach *during* it is exact arithmetic — `events.closest_approach` — and that
is what the contact distance is compared against.

The gate then had to move, because 1.12 diameters is only 0.27 in of
measurement-error budget and the estimate comes from two *filtered* paths:
smoothing rounds off the corner at contact, so a real contact never quite reads
as one. Measured — real contacts read **0.80–1.12** diameters on the three
sample clips and 0.97–1.10 on the synthetic one; near misses that ground truth
puts 1.46–1.50 diameters apart read 1.40–1.51. At 1.12 the gate sat on the
floor of that gap; at **1.25** it sits in it.

Those raw events are then grouped into **shots**, because a flat list of
"collision at t=4.12s between track 3 and track 7" is accurate but not readable.
A shot opens when a ball goes from rest to struck and closes once every ball has
settled — the referee's definition, needing no threshold of its own — and is
reported as one line: `shot 2: CUE struck -- hit #4 first -- 2 cushions --
potted #7 -- 5.4s`.

### 3.7 Output — `billiards/render.py`, `video.py`

* The overhead view is **drawn from table coordinates** rather than warped from
  camera pixels: crisp at any zoom, free to render, and it shows what the
  tracker actually believes.
* It sits in a **bar below the picture**, not in a corner of it, along with the
  status lines and the shot being played. A pool camera fills its frame with
  table — on these clips the bed covers the whole lower half — so an inset in
  any corner lands on top of the table it is describing. Nothing synthetic is
  drawn over the picture now except the tracker's own marks on the balls.
  `--overhead-inset` restores the old layout for when the output has to keep
  the source resolution.
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
* **76 tests**, including end-to-end accuracy assertions, a camera-cut test
  and a repeated-frame test.
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

Not measuring a repeated frame twice is worth another **1.3×** on top of that,
since a third of the frames in these clips no longer reach the detector at all:
21.4 → **27.3 fps** on `fedor_shot.mp4`. That only came out as a speed win
after the two frame-difference tests were moved into OpenCV — the first version
counted changed pixels with a NumPy boolean mask, which costs 19 ms a frame at
720p, more than the detection pass it was saving. Cropped to the bed's bounding
box and done with `absdiff` / `threshold` / `countNonZero`, the pair costs
1.2 ms.

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

The accuracy numbers above are from the synthetic clip, which has ground truth.
The noise numbers quoted for the real clips — tracks reported against balls
actually on the table, the swing in a ball's estimated speed between frames,
phantom events — come from `tools/run_report.py`, and every run of it is
appended to `reports/run-log.json` with the commit and what it did *not* fix:

```bash
python tools/run_report.py --ground-truth --note "what I changed"
python tools/run_report.py --show
```

That run also rewrites `results/` — the annotated video, the per-frame CSV, the
run JSON and the calibration preview for each clip — so whatever is sitting
there to look at is always what the code currently does. A stale video in an
output directory is worse than an empty one: it says nothing changed.

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
| `table.view_change_area_ratio` | Tracking pauses on a very fast pan or a strobing light (raise), or runs on through a dissolve (lower). 0 disables. |
| `detector.rim_contrast_min` | A ball almost the colour of the cloth is missed (lower), or a body part is still detected (raise). Real balls measured 23–63 on the sample clips, a hand 5–19. |
| `detector.cluster_core_coverage_min` / `cluster_rim_contrast_min` | A split blob is believed if it passes either. Balls in a dense rack are dropped (lower either), or a hand on the bed still yields balls (raise both). |
| `tracker.coast_frames_per_hit` | A ball hidden for a long time is renumbered when it reappears (raise), or brief false detections still draw trajectories (lower). |
| `table.repeat_frame_ball_areas` | A slow-rolling ball is being replayed instead of measured (lower), or a clip with sensor noise never registers a repeated frame (raise). `repeat_frame_max_run: 0` turns the whole thing off. |
| `events.contact_distance_ball_diameters` | Obvious contacts are missed (raise), or balls that clearly passed each other are reported as collisions (lower). Real contacts measured 0.80–1.12 on the sample clips, near misses 1.40+. |
| `tracker.max_speed_in_s` | Only if you film something faster than a pool break. |

Start with `python main.py calibrate <video> --save-preview calib.png`. Almost
every tracking problem is visible there first.

---

## 9. Known limitations

* A **tightly racked triangle** yields roughly 6 of 8 balls while it is static;
  all of them appear as soon as the rack separates. Adjacent balls of similar
  colour genuinely share no visible edge.
* A **stub of the cue shaft** cut off by the bed edge, with the player's body
  hiding the rest of it, is not separable from a ball resting on the cushion by
  any measurement in the pipeline: on `albin_fedor.mp4` the stub is 0.4 of a
  ball in area and 0.55 ball radii thick, and the real 8 ball on
  `fedor_shot.mp4` is 0.49 and 0.54. It survives as one intermittent track. The
  gates that would remove it would cost real balls, so it is left in.
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
* A ball that rolls into the **excluded rail margin** (`bed_margin` is 0.35 ball
  diameters, or 0.79 in) is undetectable there, so a cushion contact taken
  slowly can be missed: on `fedor_shot.mp4` the cue ball is unobserved for six
  frames precisely while it bounces off the far rail, which smears the velocity
  reversal across three coasting frames and never shows the 18 in/s jump the
  cushion test looks for. The event is missing; the trajectory through it is
  not.
* **Residual speed jitter on rewrapped broadcast footage**, about 4 in/s on
  `fedor_shot.mp4`. Dropping the duplicated frames fixes the zero-motion
  measurements, but the distinct frames that remain are still irregularly
  spaced in *true* time — the container is constant-rate at 37.5 fps while
  source frames appear at one, two or three slot intervals — and nothing in the
  file records which. `dt` from the frame index is the best estimate available.

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
  events.py      collisions, cushions, pots, balls struck
  shots.py       grouping those events into readable shots
  render.py      annotated view and synthetic overhead diagram
  video.py       input, output, CSV/JSON export
  pipeline.py    orchestration, camera-cut handling, whole-video driver
  cli.py         command line interface
tools/
  make_synthetic_clip.py   physics simulator + renderer + ground truth
  evaluate.py              MOT scoring, against the synthetic clip
  run_report.py            noise metrics on the real clips, appended to a log
legacy/          the original v1 code, kept for comparison
reports/         run-log.json: one entry per change-and-re-measure cycle
results/         rewritten by run_report.py; annotated video + data per clip
tests/           76 tests
```

`evaluate.py` and `run_report.py` answer different questions, and both are
needed. The real clips have no ground truth, so "is this output noisy?" cannot
be answered with MOTA; it is answered by self-consistency against physics —
tracks reported against balls actually on the table, how much a ball's
estimated speed swings between frames while it rolls smoothly, how far a ball
at rest is reported to move, whether the shot log says what a player saw. The
synthetic clip is then the guard against a noise fix that quietly costs recall:

```bash
python tools/run_report.py --ground-truth --note "what I changed"
python tools/run_report.py --show          # the whole history
```
