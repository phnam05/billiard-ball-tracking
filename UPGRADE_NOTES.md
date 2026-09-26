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
| MOTA | not measurable | **0.844** (0.816 on a screen-recorded-style clip) |
| Precision / recall | — | **1.000** / 0.846 |
| Median position error | — | **0.30 in** (ball radius 1.125 in) |
| Median speed error | — | **2.7%** (4.9% screen-recorded) |
| Cushion contacts found | none detected | **20 of 23**, none false |
| Colour parameters to tune | 4 arrays, per ball, per video | **0** |
| Pixel thresholds in the code | at least 6 | **0** |
| Speed (1080p source) | n/a (blocked on a key press per frame) | **26-27 fps** writing video, 60-80 without |
| Handles a camera cut | no | yes — pauses, re-finds the table |

"v1" has no notion of multiple balls or identity, so most MOT metrics are
undefined for it.

These are the numbers of 26 Sep 2026 (`reports/run-log.json`). The synthetic
clip is filmed through a physical camera behind an end rail, on the sample
broadcasts' blue-grey cloth with their ball colours, and scored on balls at
least half in view. On 23 Sep, before the cloth and ball colours were changed
(§11.5, §12.1), the same table read MOTA 0.923 / 0.932, recall 0.927, 0.21 in,
2.1% and 21 of 23 cushions. The earlier figures quoted below (MOTA 0.936, 0.150 in,
23 fps and so on) were measured on the old synthetic view, which no real camera
can produce, and are kept as they were written. The two sets aren't comparable.

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
* **131 tests**, including end-to-end accuracy assertions, a camera-cut test,
  a repeated-frame test and a screen-recorded-clip speed test.
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
| `tracker.revive_window_s` / `revive_distance_ball_diameters` | A ball that reappears is given a new number (raise), or two different balls are merged (lower). Pots are reported this late. |
| `table.source_clock` | Off, every frame is timed by the file's clock. Only engages on a file that repeats frames while balls move. |
| `table.ball_parallax` | Off, balls are placed as if painted on the cloth — 2-4 in off on a broadcast angle. |
| `events.cushion_contact_tolerance_ball_radii` | Cushion bounces far from the fitted rail are missed (raise), or balls turning near a rail are called cushions (lower). |
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
* A ball **resting in the pocket jaws** is inside the excluded region. If it
  rolls back out within `revive_window_s` it keeps its identity (see §11.4);
  if it stays there it is reported `potted` once the window has passed.
* **Cloth-coloured balls** are hard by construction — the detector looks for
  "not cloth".
* The automatic fit needs the **whole bed visible**; use `--table-corners`
  otherwise.
* Motion blur at high shutter speeds smears a fast ball into an ellipse, which
  the aspect gate can reject; the track coasts through instead.
* A **cue stick lying across the bed with a hand on it** can occasionally form a
  ball-sized blob and start a short-lived track.
* A ball **against the far cushion** appears past the far edge of the bed —
  its centre is a radius above the cloth — and so outside the region searched
  for balls. The far-rail bounce is still found from the path either side of
  it (§11.3), and a hidden ball's prediction bounces off the rail rather than
  sailing through it, but the ball itself is not seen there. Widening the
  search region to where balls *appear* fixed this on the synthetic clip and
  broke the real ones (§11.2).
* **The calibrated outline is the outline of the cloth**, and on a real table
  the cushions are clothed too: filmed from behind an end rail, the far
  cushion's face and the long cushions' tops count as bed. On those rails the
  cushion nose is up to ~4 in inside the fitted edge. Cushion detection allows
  for it; positions near those rails, and the pockets' positions, carry it.
  Fitting the nose lines themselves is the fix.
* **The source frame rate** of a screen-recorded clip is only known to about
  10% on clips this short (§11.1): which frames skipped a source frame is
  known far better than exactly how long a source frame is.
* On the retimed synthetic clip, a ball bouncing off a long rail **right
  beside a side pocket** is reported potted: for a moment, bouncing and
  heading into the pocket look the same.

---

## 10. File map

```
billiards/
  config.py      physical-unit configuration, presets, YAML/JSON
  balls.py       which numbered ball each track is, from its colour
  clock.py       the scene's clock, recovered from moving balls (§11.1)
  geometry.py    quad fitting, homography, orientation, camera, ball parallax
  table.py       cloth colour measurement, table calibration
  detect.py      ball detection, cluster splitting, colour signatures
  kalman.py      constant-velocity + friction filter, adaptive noise
  assignment.py  Hungarian assignment (+ pure-NumPy fallback)
  track.py       track lifecycle and data association
  events.py      collisions, cushions, pots, balls struck, from the raw paths
  shots.py       grouping those events into readable shots
  render.py      annotated view and synthetic overhead diagram
  video.py       input, output, CSV/JSON export
  pipeline.py    orchestration, camera-cut handling, whole-video driver
  cli.py         command line interface
  app/           the web app: library, set-up, runs, results, live (§12.2-12.3)
tools/
  make_synthetic_clip.py   physics simulator, pinhole renderer, ground truth
                           (positions, speeds, visibility, every event)
  evaluate.py              MOT, speed and event scoring against it
  run_report.py            noise metrics on the real clips, appended to a log
  robustness.py            the break under other cloths, cameras, sizes (§12.4)
legacy/          the original v1 code, kept for comparison
reports/         run-log.json: one entry per change-and-re-measure cycle
results/         rewritten by run_report.py; annotated video + data per clip
tests/           131 tests
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

---

## 11. Second pass (23 Sep 2026): the clock, parallax, events and identity

Everything above was measured on a synthetic clip that turned out to be kinder
than real footage in two ways nobody had checked, and against event counts that
nobody had ground truth for. The simulator now records every collision, cushion
contact and pot it resolves, can film a clip the way a screen recorder captures
a broadcast, and films through a real pinhole camera. Scored that way, four
problems stood out.

### 11.1 The file's clock is not the scene's — `billiards/clock.py`

The sample clips are 25-30 fps content screen-recorded at 37.5 fps. Skipping
the exact copies (§3.5a) fixed the double counting, but the frames that remain
were still timed by the slots between them. That is wrong. A ball rolling at
constant speed moves **the same distance** after a one-slot gap as after a
two-slot gap: the median ratio is 0.96, where the file's clock predicts 2.0.
Each new frame is simply the next source frame. About one step in four to
seven spans *two* source frames, because the recorder missed one, and shows
exactly twice the travel.

So each moving ball is used as a clock. Its filter predicts where it will be
after one, two or three source frames, and the count the detections decisively
agree with is taken. The source rate is estimated from those decisive frames
only. The first version also counted ties, and that was a feedback loop: the
default guess wins every tie, so the estimate drifted to 33 fps on one clip. A
version that fell back to the file's clock when unsure was worse again: two
clocks hand the filter jittery intervals, and the 95th-percentile speed error
rose from 11 to 17 in/s.

On a clip filmed like the broadcasts, speed error fell from **8.8% to 2.2%**
(median) and from 25 to 11 in/s at the 95th percentile. Real-clip speed jitter
fell by up to half, and on `albin_fedor` a missed pot was found. On
constant-rate footage the clock never engages, so nothing changes there.

### 11.2 A ball is not painted on the cloth — `geometry.raised_plane_homography`

A ball's centre is a radius above the cloth. The detector finds its silhouette,
whose centre is the image of that raised point, and the cloth homography
projected it to where the line of sight meets the cloth. That point is further
from the camera by the radius over the tangent of the viewing angle. The camera
is recoverable from the table's own homography (the focal-length constraint
that already decides the table's orientation), and on the sample broadcasts it
sits about 135 in behind the near rail and 68 in up, with a 36° lens. The error
was **2.4 in at the near rail and 4.0 in at the far one**, on every ball.

Balls are now mapped through the plane at ball-centre height. On `fedor_shot`
the cue ball's bounce off the near rail went from turning round 2.1 in short of
the cushion to 0.2 in off it. The synthetic camera had hidden this: its view
could not be produced by any real camera, and it drew balls at the image of the
spot they rested on. On the new, physical synthetic clip the tracker without
this correction scores MOTA **−0.43**.

Widening the search region to where balls *appear* (past the far bed edge)
was tried and reverted. It worked on the synthetic clip, but on `albin_fedor`
it made 30 tracks for 7 balls, because the real far edge is the top of the far
cushion's face (§9).

### 11.3 Events from the balls' raw paths — `billiards/events.py`

The filter turns a corner over two or three frames. By the time its velocity
visibly reverses, the ball is several inches off the cushion it hit, so the old
cushion test found **2 of 23** contacts on the synthetic break. Events now come
from the raw detections:

* **Cushions:** for each rail, each raw step is classed as toward or away from
  it. A change from toward to away, within reach of the rail and with no other
  ball near, is a contact. Where the ball was not seen at the bounce, the two
  lines either side of it are extended until they meet. A corner fit (three
  samples in, three out, best split, committed only once the next split has
  been tried and lost) adds a few more.
  - An earlier version accepted the first split that differed, which put the
    corner a frame early, with the outgoing line straddling the bounce.
* **Collisions:** two balls' paths turning a corner at the same moment, a
  ball's width apart and closing, or one ball's corner beside a ball that did
  not visibly turn.
  - The sampled closest-approach test (§3.6) stays: it finds contacts where the
    ball is potted or hidden too soon after for its path to turn a corner.

Cushions found went from **5 to 49 of 55** across the synthetic clips, with
no false ones, and `fedor_shot` now reports its four cushions in order. Event
ground truth ignores collisions between balls that were already touching (a
static rack passing momentum on the break), which no camera can see.

### 11.4 A ball that comes back is the same ball — `billiards/track.py`

A track that dies now waits `revive_window_s` in limbo. A new detection of the
same colour near where it vanished, or further along its path if it was
rolling, is that ball, and its death, and any pot, is withdrawn. Pots are
therefore reported late, at the frame the ball vanished, and the shot log files
late events under the shot they happened in.

On `albin_fedor` the cue ball stopped in the pocket jaws after knocking the 4
in. That had been logged as a scratch and turned the cue ball into "#10" for
the rest of the clip. Both are gone. Two related fixes:

* The cue-ball and 8-ball roles are held through coasting and through
  shadows, which had dropped a cue ball's white fraction from 0.77 to 0.41.
* An unseen ball's prediction bounces off a rail it reaches, unless it is
  heading into a pocket.

### 11.5 The benchmark changed

The synthetic clip is now filmed through a physical pinhole camera behind an
end rail. It has raised cushions with cloth faces and a venue of grey carpet
and sponsor banners, and every ground-truth row records how much of the ball is
in view. Balls less than half visible are ignored when scoring, as MOT
benchmarks ignore occluded targets. **Numbers before and after 23 Sep 2026 are
not comparable**, and the run log says so where they change.

---

## 12. Third pass (26 Sep 2026): an app, live tracking, and other footage

### 12.1 Loose ends from the 23 Sep evening

The evening of 23 Sep added ball numbers (`billiards/balls.py`), gave the
simulator the broadcasts' own ball colours and cloth, and began fitting the
table to the cushion noses (`table.refine_to_cushion_noses`). It stopped with
two tests failing and nothing written up. Picking it up:

* **Phantom cushions.** `fedor_shot` reported a cushion 39 in from any rail,
  and `fedor_jump` one 24 in away. A ball rolled toward a rail, stopped short,
  and was knocked away seconds later. Steps too small to class leave a ball's
  "approaching" state alone, so the knock read as the far side of a bounce. An
  approach now expires once the ball has been *seen* all but still for
  `events.cushion_approach_max_age_s` (0.6 s). A plain age limit was tried
  first. It also dropped a real bounce on `albin_fedor`, where the cue ball
  spends 0.6 s hidden in a pocket's jaws, and an unseen ball proves nothing
  about stopping. Both clips now report 4 cushions, as the video shows; the
  synthetic scores did not move.
* **The clock ran away.** On the 4-second screen-recorded test clip it read
  **57.9 fps** off 25 fps content, and on `fedor_shot` 41.1 fps from a
  37.5 fps file: more source frames than the file has slots, which is
  impossible. The cause is the filters' lag. Whether a step spans one source
  frame or two is judged against a ball's velocity, and that velocity is in
  the time base the filter was fed. When the rate estimate rises, for the few
  frames a filter takes to catch up, a one-frame step reads as two, the
  estimate rises further, and so on. Three changes:
  1. Every ball's velocity is re-expressed in the new clock whenever the
     estimate changes (`BallKalman.rescale_time`).
  2. The rate is bounded by counting frames: no more than the file's rate, no
     less than the new-frame rate, and no more than that over 0.65.
  3. It snaps to a broadcast standard within 5% (was 3%), because decisive
     frames lean toward short gaps and read 25 fps content 3-4% fast. The
     standards' 5% windows do not overlap.

  Fitting a period to the measured intervals instead was tried and dropped.
  Positions alone cannot tell a ball twice as fast filmed half as often, so
  every measured interval simply echoed the period the filter had been fed.
  Result: 25.0 fps on the test clip, and speed error on the report's longer
  screen-recorded clip went from 5.2% to 4.9%. Real-clip jitter rose slightly
  (`fedor_shot` 2.65 to 3.16 in/s). The three real clips still read 30.0,
  23.976 and 33.3 fps though they come from one broadcast, so their true rate
  is not known.
* **The far cushion, again.** Searching the band past the far edge, now that
  the edge is at the nose, raised synthetic recall 1.6 points and removed
  `fedor_shot`'s extra track. On `albin_fedor` the black 8 against the far
  rail ran into the dark line under the nose and split in two. That stayed
  true after two mitigations: nothing new may start in the band, and a band
  detection overlapping a bed ball is dropped. It is kept as
  `detector.search_raised_bed`, off.
* **Test bars** re-set to the harder benchmark, with measured values beside
  them: recall 0.797, MOTA 0.783 on the constant-rate fixture. Most of the
  misses are the blue stripe, whose band the blue-grey cloth mask takes for
  cloth (recall 0.31).

### 12.2 The app — `billiards/app/`

`python main.py app` serves a page on `127.0.0.1:8765` and opens it. It uses
only the standard library (`http.server`), so it needs nothing the tracker does
not. The page is plain HTML, CSS and JavaScript with no build step and no CDN,
because this network blocks some of them.

| Module | Does |
|---|---|
| `workspace.py` | The library (folders scanned, files added, uploads), per-video settings, run folders, probed metadata and thumbnails, all under `billiards-workspace/`. A video is known by a hash of its path. |
| `jobs.py` | Runs in the background, one at a time by default: progress, the latest annotated frame as a JPEG, events so far, cancel. At the end it writes `viewer.json`, every ball's path arranged for the page, with the table and each ball's colour. |
| `preview.py` | The set-up check: calibration (cached per video and settings) and detection on one chosen frame, with plain-language warnings, and the cloth mask. |
| `live.py` | Live sessions (§12.3). |
| `server.py` | The routes, including `Range` for seeking in video, MJPEG for live pictures, and uploads. It refuses requests whose `Host` or `Origin` is not the app, so a web page open in the same browser cannot drive it. |

`pipeline.run` gained the hooks this needs: `on_frame`, `should_stop`,
`annotate` and `writer`. It also writes cloth measured inside corners placed
by hand (§12.4).

**Video a browser plays.** OpenCV's pip wheels write H.264 only through
Windows Media Foundation (about 50 Mbit/s, ignoring the quality setting) or
not at all, and their VP8 writes at about 40 fps. `video.browser_codec()`
picks, best first: `ffmpeg` from `imageio-ffmpeg` (libx264 at about 180 fps,
CRF 20), OpenCV's `avc1`, then `VP80`, each tried once by writing and reading
back. Failing all three it writes `mp4v`, and the page shows it frame by frame.

**The results page** plays the tracked video with a top-down view drawn from
`viewer.json`, kept in step through `requestVideoFrameCallback`. It has a
timeline of shots and events, lists of shots, events and balls, and a speed
chart. Events are named by the ball's final label ("cue ball hit the 6"),
because the event log only carries track ids. A diagram colours balls by
number in the set the tracker chose (`summary.ball_set`), since measured
colours are dull. `#/run/<id>?t=4.7&tab=events&ball=6` opens it at a moment.

### 12.3 Live — `billiards/app/live.py`

Frames do not wait. A reader thread keeps only the newest frame and its number
in the stream, and the tracker takes whichever is newest when it is ready. When
tracking is slower than the camera, frames are skipped, and the skipped numbers
become gaps its clock sees. The table is found from 12 frames over the first
second and a half, or from corners placed by hand on a snapshot, and the search
retries until it succeeds. A recording repeats the last picture over skipped
frames, so it keeps the stream's clock. Stream addresses open with 8 s timeouts.
Without them a wrong address takes FFmpeg half a minute to give up on.

Measured on `fedor_shot` replayed at its own pace: 38 fps in, 26-29 fps
tracked, about one frame in four skipped. The shot log read "CUE struck, hit
the 2 first, 3 cushions, potted the 2" against 4 cushions from the file, the
cost of the skipped frames. Tested end to end on a synthetic clip. Not tested
with a real camera: the work computer's webcam was not turned on.

### 12.4 Footage unlike the sample clips — `tools/robustness.py`

The three sample clips are one venue, one camera angle and one cloth. The
robustness tool renders the same break under the conditions other footage
brings, tracks each with default settings, and scores it against ground truth.
It writes `reports/robustness.json` and `results/robustness.png`. Two things it
found were the scoring's fault, not the tracker's:

* **A pool table is symmetric.** Which corner the tracker calls (0, 0) depends
  on where the camera stands, and from a long rail, the ceiling or a corner it
  was a mirror of the simulator's. Scored without allowing for that, three
  views that track well read MOTA -0.4 to -0.75. The tool (and
  `evaluate.best_symmetry`) now scores under whichever of the four symmetries
  fits.
* **The simulator drew its shadow lines 2 px at every size.** Scaled from
  1080p to the tracker's 1280 px, the line under each cushion's nose became a
  sub-pixel trace, the clothed cushion tops ran into the bed, and the outline
  was fitted around both: MOTA -0.04. The line now scales with the picture.
  The weakness it exposed is real, though: a faint nose line lets the outline
  take in the cushion tops, and a per-rail offset cannot correct the skewed
  quadrilateral that results.

Measured 26 Sep (4 s of play each, default settings; `reports/robustness.json`
holds the latest run). Each row differs from the first in one respect:

| Variant | MOTA | Recall | Position | Speed | Numbers right | Cushions |
|---|---|---|---|---|---|---|
| baseline: end camera, blue-grey cloth, 720p, 30 fps | 0.81 | 0.81 | 0.28 in | 2.5% | 91% | 16/20 |
| green cloth | 0.87 | 0.87 | 0.24 in | 2.2% | 97% | 18/20 |
| tournament-blue cloth | 0.87 | 0.88 | 0.27 in | 2.4% | 82% | 19/20 |
| burgundy cloth | 0.86 | 0.86 | 0.30 in | 2.4% | 95% | 19/20 |
| camel cloth | 0.74 | 0.74 | 0.29 in | 2.1% | 94% | 19/20 |
| grey cloth | no table found (refused); 0.79 with corners placed by hand | | | | | |
| camera across from a long rail | 0.72 | 0.72 | 0.39 in | 2.4% | 87% | 15/20 |
| camera on the ceiling | 0.88 | 0.89 | 0.41 in | 2.2% | 77% | 19/20 |
| tripod at a corner, table small | 0.57 | 0.72 | 1.87 in | 6.0% | 76% | 18/20 |
| 854x480 | 0.77 | 0.80 | 0.41 in | 3.9% | 72% | 19/20 |
| 1920x1080 | 0.79 | 0.79 | 0.29 in | 2.4% | 89% | 17/20 |
| filmed at 60 fps | 0.80 | 0.82 | 0.32 in | 2.5% | 85% | 11/15 |
| screen-recorded broadcast | 0.80 | 0.80 | 0.22 in | 4.7% | 88% | 12/19 |
| the broadcasts' ball set | 0.82 | 0.82 | 0.27 in | 2.5% | 89% | 18/19 |

The 480p row is from after the nose search was deepened (below); before it, and
after the simulator's lines were scaled, 480p read MOTA -0.03: the far cushion's
top and face merged with the bed, 6.5 in past the nose, and the search only
looked 6 in in. `table.cushion_nose_search_in` is now 10 in, which puts the
480p outline within 1.1 in of true, the same as at 720p, and leaves the
synthetic benchmark unchanged.

Three fixes came out of it:

* **Grey cloth.** A nearly neutral cloth has no hue to window on, and its
  pixels were rejected as "too grey to be cloth". Searched for automatically,
  the most saturated thing in view, a blue banner, was taken for the table,
  and every ball came out 6 px across. Now:
  - A table on which a ball would be under 6 px across (`table.min_ball_radius_px`)
    is refused, with a message that says why.
  - With corners placed by hand, the cloth is measured inside them. If most
    of that bed is unsaturated, it is modelled with no hue: unsaturated, and
    about this bright, with a narrow window above its brightness so the
    ivory cue ball stays out (`ClothModel.neutral`). Grey cloth with its
    corners placed: MOTA 0 before, 0.79 after, against the benchmark's 0.81.
* **Hand-placed corners measure the cloth inside them**, for any cloth: in a
  wide shot the most common saturated colour can be the floor or a banner.
* **A frame that will not decode** in the middle of a file is skipped, up to
  eight in a row, instead of ending the run. Paths with characters outside the
  system code page are opened through their Windows short name on OpenCV
  builds that cannot open them directly.
