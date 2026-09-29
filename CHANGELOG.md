# Changelog

What changed, newest first. The reasons behind each
change, with the measurements, are in [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md).

## 2026-09-29

### Fixed (afternoon): a ball in a pocket's jaws, jump shots, and stickier numbers

- **A ball hanging in a pocket's jaws is found**: inside the disc blanked out
  round each pocket, something neither cloth nor as dark as the hole is a
  ball if the ball model is at least 0.98 sure (`detector.pocket_ball_p`).
  `albin_fedor`'s 4, potted by the first shot, was never seen.
- **A ball that vanishes in a pocket's mouth is potted there**, not coasted
  on along the rail; **pots are dated to the ball's last sighting**; an event
  just before a shot that was noticed late belongs to it.
- **A ball passed over is not hit**: a contact with a ball at rest is
  reported only once that ball moves (or drops); the jump over the 6 had read
  "hit the 6 first".
- **A ball's number is harder to take away** (`balls.model_stickiness` 0.4 →
  0.8): the US Open's 7 had turned into the 1 for 8 s.
- Measured: `albin_fedor` "potted the 4"; US Open named wrong 1% → 0%;
  Premier League 0.813 → 0.797 (two phantoms after dissolves), its pots right
  6 → 7 of 8; false contacts down on the synthetic break and 12 robustness
  variants. Tests: 183.

### Fixed: a ball on the far cushion, numbers after a pot, and the shot log

- **A ball resting against the far cushion from the start is tracked** when
  the ball model is at least 0.98 sure it is a ball
  (`detector.raised_band_start_p`). Past the bed's far edge a detection used
  only to follow a ball already tracked; the tripod answer key's 1 was
  missed on 12 of its 16 keyframes.
- **A detection on a ball at rest is that ball**, whatever its colour, if it
  is within half a ball of it: two balls cannot overlap. A ball touching
  another, split out of their blob, had started a second track on top of it.
- **A potted ball's number is held until a new rack**, which shows itself by
  more balls on the table than can be left of this one. Freed at once (28
  Sep), a look-alike took it: the tripod's far 1 became the potted 9.
- **Shots end after 0.4 s of play at rest, not 0.4 s of the file's frames**
  (on a broadcast that repeats every other frame that was 0.8 s, and shots
  merged); **a ball trembling in place is not moving**, and **a shot in which
  no ball went two ball widths, and nothing was potted, is dropped** (a rack,
  split a pixel off each frame, opened shots before the break).
- **A strike between two sightings is caught**: a ball at rest seen next
  more than a ball's width away within 0.15 s is taken at that speed.
- **Balls far apart setting off together, with nothing moving, are not
  struck**: that is the picture changing (a dissolve to another camera), not
  play. It removed a phantom pot on the Premier League minute.
- **`tools/real_eval.py` scores the shot log**: shots found (within 1.5 s),
  extra, and pots right, against each key's shots. The Premier League key
  lists a shot already rolling at frame 0.
- Measured: answer keys US Open 0.941 → 0.952, Premier League 0.791 → 0.813,
  ceiling camera 1.000, tripod 0.742 → 0.924; named wrong on the two
  CCTV-style clips 12% → 0% and 18% → 12%; shots found 3/8 → 8/8 (Premier
  League), 2/3 → 3/3 (tripod), ceiling camera's 3 extra → 0. Synthetic breaks
  unchanged. Still: dissolves open 4 extra shots, replays are tracked, the
  side view is not. Tests: 178.

## 2026-09-28

### Added (evening): answer keys for real footage, and a learned ball check

- **Answer keys for four real clips** (`tools/truth/`): the 2026 US Open 3
  min, the 2026 Premier League final minute, a ceiling camera over a club
  table and an amateur 8-ball match from a tripod, marked by hand on
  keyframes. **`tools/real_eval.py`** scores a run against them (balls found,
  tracks that are balls, numbers right and wrong, ids per ball, phantoms, what
  was drawn in replays and close-ups) at any checkout; `run_report.py --real`
  logs it. The US Open "got worse" (25 → 66 balls) was a counting artefact:
  the older code restarted its ids at every camera change. Against the key,
  `49446ee` scored 0.83 there against 0.35 before it; only its ball numbers
  were worse.
- **A learned check on every proposed ball** (`billiards/ballnet.py`,
  `billiards/models/ballnet.onnx`, 288 KB): is it a ball, the cue ball, which
  colour, a stripe. It drops chalk, knuckles and pocket shadows, keeps balls
  the shape tests turned down (a ball welded to its shadow, balls in a clump),
  and names the balls in place of the colour palette. It runs through OpenCV,
  so nothing new is needed to use it; `tools/ballnet_data.py` and
  `tools/train_ballnet.py` (PyTorch) rebuild it from `tools/ballnet/`.
  `detector.ball_model: off`, or `BILLIARDS_BALLNET=off`, turns it off.
- **Someone walking past the lens is no longer a cut**: when the rest of the
  bed is the picture it was and still cloth, tracking goes on and the hidden
  balls coast.
- **A ball keeps a colour per camera**, and is found again after a cut to a
  camera that shows it differently.
- **A lost ball waits 20 s** for a ball like it where it was
  (`tracker.lost_revive_window_s`).
- **Grey cloth measured with no colour is told apart without hue**: a ceiling
  camera's table had been fitted 3 in short of both end cushions.
- **Balls against the far cushion are searched for by default**
  (`detector.search_raised_bed`, the app's *far cushion* option).
- Ball numbers: a potted ball's number is free again with the model (a new
  rack); the ball set's evidence is kept for about 40 s, not 2; brown and
  orange count for each other at half weight; a stripe's caps still count.
- Measured: answer keys US Open 0.83 → 0.94, Premier League 0.71 → 0.79,
  ceiling camera 0.68 → 1.00, tripod 0.49 → 0.74. Synthetic break MOTA 0.844
  → 0.910, screen-recorded 0.816 → 0.917, every robustness variant higher.
  `albin_fedor`'s missed pot of the 5 is now reported. Costs: more ID
  switches on the screen-recorded break (2 → 10), speed error 4.9% → 5.5%
  there, `fedor_jump` 3 cushions for 4, about 10% slower. Replays are still
  tracked as play. Tests: 167.

### Changed (evening): `python main.py` opens the app

- **`python main.py` with no arguments opens the app**, as `python main.py
  app` does. It used to print the usage and exit, so an IDE's Run button
  (PyCharm's "main" configuration passes no arguments) never opened the app.
  `python main.py clip.mp4 ...` and the other commands are unchanged.

### Fixed (late afternoon): the same balls across a broadcast's cuts

- **A ball keeps its identity when the broadcast cuts away and back**, to
  the same camera or another. At a cut every ball is set aside, rolled on
  by its motion model while the table is out of view, and re-found when it
  is back: near where it should be, or anywhere by an unambiguous colour.
  Balls not found in 4 s of the table in view are lost, or potted if they
  were heading into a pocket. Before, every ball became a new one, and one
  that coasted near a pocket was reported potted.
- **The tracker is never rebuilt.** It used to be rebuilt whenever the
  table's outline looked different, including every 5 s on grey cloth whose
  cushion tops come and go from the outline. Track ids restarted at 1 each
  time, so `tracks.csv` had different balls under one id. Ids are now unique
  for the whole clip.
- **Camera views are remembered**, and a view seen before is recognised from
  the picture: its bed is cloth and the band round its rails is not. A new
  view gets its cushion noses fitted, and is turned or **mirrored** to put the
  balls where they were. A camera across a long rail used to see the table
  mirrored relative to the end-rail camera (`TableModel.mirrored`).
- **The camera is followed when it pushes in or pans** (`billiards/camera.py`):
  features round the table are matched to a keyframe of the view, and the
  table is carried along by the fitted motion. Before, the outline was
  refitted, which a player over the near rail spoils.
- **The table re-check ran on no frame at all on 60 fps broadcasts** (it ran
  on frame numbers that were all repeats). It now runs every 30 measured
  frames (`table.recalibration_interval`, was 150), sooner when the bed stops
  looking like cloth.
- A median of several frames' outlines could cross itself (corners in two
  orders). Outlines that cut a corner are no longer evidence. Outlines fitted
  while tracking use a half-size mask that keeps the line under the cushion
  nose (it had come out 6 in too big).
- A clip that opens on something other than the table starts with tracking
  paused.
- `tools/robustness.py` *cuts* variant (`make_synthetic_clip.py --cuts
  broadcast`), and `evaluate.py` reports **ids per ball**. On that clip: MOTA
  0.404 → 0.782, precision 0.762 → 1.000, ids per ball 2.3 → 1.2.
- Real minutes, balls reported for at most 10 on the table: Mosconi Cup 91 →
  44, Premier League 129 → 33, UK Open 48 → 26. Speed 35.1 → 31.0, 42.6 →
  36.9 and 22.4 → 23.8 seconds per minute of footage.
- New `TrackerConfig` settings: `reclaim_distance_ball_diameters`,
  `reclaim_colour_ratio`, `reclaim_window_s`. `run.json` adds `views`,
  `camera_moves` and `tracks_reclaimed`.

### Fixed (afternoon): footage from other venues

- **The table is found when the floor is a stronger colour than the cloth.**
  A 2026 US Open clip tracked nothing: the cloth was taken to be the
  commonest saturated colour, and that was the royal-blue floor, not the
  blue-grey cloth. Now the commonest colours (by hue and saturation) and
  greys are each tried, and the one whose outline behaves like a table is
  kept (`table.choose_cloth`): it agrees from frame to frame, the colour
  fills it, and it doesn't run off the picture. The usual estimate is kept
  unless another scores 1.25× higher.
- **Grey cloth is found automatically**, with no corners placed by hand
  (simulated: MOTA 0.79, the same as with corners placed).
- **A highlight reel's several cameras**: calibration uses the biggest group
  of frames that agree, one camera's view, instead of a median of all of
  them (the 2025 UK Open cuts between three).
- **An outline that runs off the picture is never taken for the table**,
  after a cut either (a UK Open close-up had its score bar's ball icons
  tracked). When a camera zooms in until the cloth runs off the picture,
  tracking pauses until the whole table is back.
- Robustness (simulated): new *blue-floor* 0 → 0.81 and *green-red-floor*
  0 → 0.86 variants (`make_synthetic_clip.py --floor`); grey cloth "table not
  found" → 0.79. The other 13 variants, the synthetic benchmark (0.844 /
  0.816) and the sample clips are unchanged.
- `tools/venues.py`: one real minute from each of 11 venues, tracked the way
  the app tracks (`reports/venues.json`, `results/venues.png`). The table is
  now found at all 11. Downloads go to `.cache/`, which is not committed.
- `calibration.cloth_candidates` in `run.json` lists every colour tried and
  its score.
- 3 tests; 152 pass.

### Added

- **Track a YouTube link, a few minutes at a time** (`billiards/fetch.py`,
  `billiards/app/downloads.py`). A match on YouTube is often an hour long, and
  tracking all of it takes longer than watching it, so:
  - In the app, **Add videos** starts with a link box. Pasting a link looks it
    up in about 3 s and shows the title, length and picture size, plus how
    long tracking the whole video would take on this computer. For a video
    over 10 minutes it suggests a 5-minute part.
  - You pick the part (`20:00` to `25:00`, or the *3 min* / *5 min* / *whole
    video* buttons), and the estimate for that part updates as you type. A
    link made with YouTube's *Share → Start at* (`&t=1200`) fills in the start.
  - **Download and track** fetches only that part, at 720p, H.264, no sound,
    into `billiards-workspace/downloads/`. It is queued for tracking as soon
    as it arrives. The library shows the download with its progress, and it
    can be stopped (nothing is left behind) or tried again.
  - The game is read from the title (*10-Ball* → balls 1–10), and can be
    changed.
  - From the command line: `billiards "<link>" --start 20:00 --end 25:00 -o
    out.mp4` prints the estimate, downloads the part, then tracks it.
- The estimate is a **range**: the fastest to slowest of this computer's last
  5 runs. Until there are 3 runs, the pace measured on a broadcast (35.4
  frames a second through the app) is part of the range.
- Library cards of untracked videos say about how long tracking will take.
- `track --numbers 1-10` (balls in play) and times as `m:ss` / `h:mm:ss` for
  `--start` / `--end`.
- `yt-dlp[default]` is an optional dependency, in `requirements.txt` and the
  `app` / `full` extras. When it is missing, the app says how to install it.
- 18 tests (`tests/test_fetch.py`), with yt-dlp played by a script so no test
  needs the network. 149 pass.

### Fixed

- Durations such as 419.6 s read "6 min 60 s" in the app.

### Measured (no tracker change, so `results/` was not regenerated)

- On 20:00–24:00 of the example final (720p, 60 fps, 32 camera changes),
  the 4 minutes tracked in 407 s through the app (35.4 fps), 494 s from the
  command line with `-o --no-overhead` (29.2 fps), and 674 s with `-o` and
  the diagram drawn in (21.5 fps). The whole 61-minute video would take about
  1 h 45 min in the app. Downloading the 4-minute part took 17 s.

## 2026-09-27

### Changed

- **The app says what to do next.** It was unclear where to start and what to
  click: each video had four equal buttons, the blue one (*Track*) re-ran a
  video that already had results, and clicking the picture opened the set-up.
  - The library opens with a guide, *① Add a video → ② Track it → ③ See the
    results*, with the current step lit and one sentence on what to do now.
  - Each video has one blue button for its next step: **Track**, then **Watch
    progress** while it runs, then **See results**. Clicking the picture does
    the same (but never starts a run). *Track again*, *Change set-up* /
    *Check the table first* and *Remove* are small links; *Remove* asks first.
  - **Upload** and **Add footage** are one **Add videos** dialog, with
    choosing files first.
  - **Runs** is called **Results** throughout.
  - The set-up page says what to check and offers the last results; the
    results page says to press play or click an event, and its video, CSV and
    JSON buttons are one **Download** menu. *Set up & track again* is
    *Change set-up*.
  - Help, README and `docs/images/app_results.png` describe the new layout.

### Fixed

- The Live page showed the word "null" and a broken-picture icon before a
  session was started.

## 2026-09-26 — commit `0fe9fc9` (with 23 Sep)

### Added

- **The app** (`billiards app`, package `billiards/app/`): a local web page for
  the whole tool, standard library only, no build step, no CDN.
  - Library of footage: folders, single files, drag-and-drop uploads,
    thumbnails, last result per video.
  - Set up: the table outline, pockets and detected balls drawn over any
    frame, warnings in plain words, the cloth mask, corners placed by hand by
    dragging, and per-video settings (table, ball set, balls in play, time
    range, processing size).
  - Runs in the background with live progress and preview; stop keeps what
    was tracked.
  - Results: the tracked video with a synced top-down view, a clickable
    timeline of shots and events, shot/event/ball lists, a speed chart,
    downloads, units (km/h or in/s), keyboard control, and links that open a
    moment (`#/run/<id>?t=4.7`).
  - Listens on 127.0.0.1 only and refuses cross-site requests.
- **Live tracking** (`billiards/app/live.py`): a webcam, a network stream
  (RTSP/MJPEG/HTTP), or a library video replayed at its own pace. The newest
  frame is always taken, so a slow tracker skips frames instead of falling
  behind. The table is found from the first ~1.5 s, or from corners placed on
  a snapshot. Sessions can be recorded as runs.
- `video.browser_codec()` / `FfmpegSink` / `open_sink`: annotated video a
  browser plays (H.264 via `imageio-ffmpeg`, else OpenCV's `avc1` or `VP80`).
- `RunOptions.on_frame`, `should_stop`, `annotate`, `writer`; the run summary
  gains `stopped_early` and `ball_set`.
- `tools/robustness.py`: the break rendered with 6 cloths, 4 camera positions,
  3 resolutions, 60 fps and a screen recording, scored against ground truth;
  writes `reports/robustness.json` and `results/robustness.png`.
- Simulator: cloths `blue`, `red`, `tan`, `grey`; cameras `overhead`, `corner`.
- `evaluate.best_symmetry` / `symmetric_tracks`: score under whichever
  symmetry of the table the tracker's corner (0, 0) came out as.
- Grey (neutral) cloth, when the corners are known (`ClothModel.neutral`).
- `table.min_ball_radius_px` (3.0): a "table" on which a ball would be under
  6 px across is refused rather than tracked.
- `imageio-ffmpeg` in `requirements.txt` and a `[app]` extra; the app's page
  files ship as package data.
- 13 app tests (`tests/test_app.py`: workspace, every API route, the
  cross-site guard, a run end to end, stop, and live from a file) and 3 unit
  tests for grey cloth, the too-small table and a bad frame.

- `detector.search_raised_bed` (off by default): also search the band past
  the bed's far edge where a ball against the far cushion appears. Only a
  ball already being tracked is followed there; nothing new is started in it,
  and a band detection overlapping a bed ball is dropped. Off because on
  `albin_fedor` the 8 ball, against the far rail, still splits into the dark
  line under the cushion's nose (8 tracks for 7 balls). In the app it is a
  per-video check box, *Look for balls against the far cushion*.
- `BallKalman.rescale_time` / `peek_many`, `MultiObjectTracker.rescale_time`.
- 3 tests: a ball seen standing is no longer approaching a rail; the
  clock's rate cannot exceed the file's; velocities follow a change of clock.
  With the app's and robustness tests, 19 new today, 131 in total.

### Changed

- **Scene clock** (`billiards/clock.py`): the source rate is bounded by
  counting -- no more than the file's own rate, no less than the new-frame
  rate, and no more than that over 0.65 -- and when the estimate changes,
  every ball's velocity is re-expressed in the new clock. Without this, a
  filter's lag after a change made one-frame steps read as two and the
  estimate ran away: 57.9 fps off 25 fps content on the test clip, 41.1 fps
  off `fedor_shot`, whose file is 37.5 fps. Rates now snap to a broadcast
  standard within 5% (was 3%).
- New setting `events.cushion_approach_max_age_s` (0.6 s).
- `table.cushion_nose_search_in` 6 → 10 in. At 480p the far cushion's top and
  face merged with the bed 6.5 in past the nose, beyond the search, and the
  480p synthetic clip scored MOTA -0.03 (now 0.77). The synthetic benchmark
  is unchanged.
- Test bars re-set for the 23 Sep evening benchmark, with measured values in
  the comments.
- `config/default.yaml` regenerated; it was missing the 23 Sep evening
  settings (`fit_cushion_noses`, the `balls:` section), then
  `min_ball_radius_px` and the 10 in nose search.
- README and `UPGRADE_NOTES.md` §1: accuracy at the current benchmark (MOTA
  0.844 / 0.816, `reports/run-log.json` 26 Sep 17:21), a table of how other
  footage tracks (`tools/robustness.py`), the limitations it found, 131 tests,
  and a screenshot of the app's results page (`docs/images/app_results.png`).

### Fixed

- A run no longer ends at a frame that will not decode: up to 8 in a row are
  skipped while the file has more.
- Videos whose path has characters outside the system code page open on
  OpenCV builds that could not open them (through the Windows short name).
- Corners given by hand (`--table-corners`, or the app) now measure the cloth
  inside them, instead of over the whole frame, where a floor or banner of a
  similar colour could win.
- The simulator's shadow lines scale with the picture; at 1080p they had
  thinned to nothing once scaled down, and the synthetic 1080p clip scored
  MOTA -0.04 (now 0.79).
- **Phantom cushions** 24-39 in from any rail (`fedor_shot` frame 303,
  `fedor_jump` frame 265): a ball that rolled toward a rail, stopped short
  and was knocked away seconds later "bounced" off it. A ball seen standing
  for `cushion_approach_max_age_s` is no longer approaching a rail; a ball
  *unseen* for that long, in the jaws or under a hand, still is.
  `fedor_shot` and `fedor_jump` now report 4 cushions each, as in the video.

## 2026-09-23 — in commit `0fe9fc9`

Numbers against synthetic ground truth changed meaning today: the synthetic
clip is now filmed through a physical camera. See *Changed* below and
`UPGRADE_NOTES.md` §11.

### Added

- **Scene clock** (`billiards/clock.py`): on screen-recorded broadcasts the
  file's clock is not the scene's. Each new frame is the next source frame,
  whatever the slot gap, and sometimes two source frames on when the recorder
  missed one. The moving balls now decide which, and the source frame rate is
  estimated from decisive frames only. It engages only on files that repeat
  frames while a ball is moving, so constant-rate footage is untouched.
  - New settings: `table.source_clock`, `source_clock_min_moving_repeats`,
    `source_clock_max_skip`, `source_clock_min_step_ball_radii`.
  - The run summary gains a `clock` block.
- **Ball parallax**: the camera is recovered from the table's homography
  (`geometry.raised_plane_homography`, `TableModel.camera`), and balls are
  placed through the plane at ball-centre height.
  - New methods: `ball_image_to_table` and `ball_table_to_image`.
  - New setting: `table.ball_parallax`.
- **Path-based events** (`events.py`):
  - Cushions from motion toward a rail turning into motion away
    (`_rail_reversal`).
  - Corners in a ball's raw path (`find_kink`), paired into collisions, or a
    single corner beside a ball that didn't visibly turn.
  - New settings: `events.cushion_contact_tolerance_ball_radii`,
    `cushion_pocket_clearance_ball_diameters`, `kink_pair_window_s`,
    `kink_pair_distance_ball_diameters`, `kink_refractory_s`.
- **Track limbo** (`track.py`): a dead track waits `tracker.revive_window_s`.
  If its ball reappears nearby (or further along its path) with the same
  colour, it gets its identity back and any pot is withdrawn.
  - Pots are dated to the frame the ball vanished.
  - The shot log files late events under the shot they belong to.
  - New settings: `revive_window_s`, `revive_distance_ball_diameters`.
  - The run summary gains `tracks_revived`.
- **Simulator** (`tools/make_synthetic_clip.py`):
  - `--events` writes every collision, cushion contact and pot the physics
    resolved.
  - `--container-fps` / `--drop-rate` / `--capture-jitter` write a clip the
    way a screen recorder captures a broadcast.
  - `--camera end|side` chooses a physical pinhole camera.
  - The ground truth gains a `visible` column: how much of each ball is in
    view.
- **Evaluator** (`tools/evaluate.py`):
  - Speed error against ground truth.
  - Event recall and precision (`--events-gt`, `--run-json`).
  - Visibility-aware scoring (`--min-visible`): balls less than half in view
    are ignored.
- **Run report** (`tools/run_report.py`):
  - Ground truth now includes speed error and events.
  - A second, broadcast-style synthetic clip is scored.
  - Clips carry `detectable_pots` alongside `real_pots`.
- `legacy/README.md`, crediting Stuart Grieve's 2015 *PoolTable* project, and
  his author headers restored in `legacy/PoolTable.py` and `legacy/Indexer.py`.
- 12 tests (88 total), including end-to-end speed accuracy on a retimed clip,
  path-based events, parallax against a pinhole camera, limbo, and
  visibility-aware scoring.
- *23 Sep evening* (written up 26 Sep from the session log):
  - **Ball numbers** (`billiards/balls.py`, `balls:` settings): each coloured
    ball is named 1-15 by one optimal assignment across the whole table,
    with the venue's lightness and saturation fitted from the table itself.
    Two ball sets (`standard`, and `tv`: 4 pink, 5 purple, black-capped
    stripes), chosen by accumulated evidence. Labels, the CSV's `number`
    column and the shot log use them ("potted the 5").
  - Colour signature measures each ball's own colour and a stripe score.
  - `table.fit_cushion_noses`: the fitted outline is moved in to the cushion
    noses (the clothed cushion tops look like bed).
  - Simulator: the broadcasts' ball colours and cloth, ivory whites, black
    caps, rolling number circles, raised cushions; ground truth gains a
    `number` column.
  - 24 tests (112 total), 2 of them left failing on the harder benchmark.

### Changed

- **The synthetic benchmark is physically realistic, and its numbers are not
  comparable with earlier ones.**
  - The old view could not be produced by any pinhole camera, and it drew
    balls as discs painted on the cloth. It is now filmed from behind an end
    rail through the camera recovered from the sample broadcasts.
  - The table now has raised cushions with cloth faces, and the venue has grey
    carpet and banners.
  - New figures: MOTA 0.923 / 0.932 (constant-rate / screen-recorded), 0.21 /
    0.17 in position error, 2.1% / 4.1% speed error.
- Cue-ball and 8-ball roles are held through coasting, and through shadows
  that dim a ball, instead of being dropped.
- An unseen ball's prediction reflects off a rail it reaches, unless it is
  heading into a pocket.
- Test thresholds re-set for the new benchmark, with the measured values
  recorded beside each one.
- New demo image, from the new synthetic view.
- README: accuracy, how-it-works and limitations brought up to date. The
  sample shot log is now real output (the old one was phantom-era).
- `DIARY.md` is one table row per step, 308 → 159 lines, and the rule in
  `CLAUDE.md` says so.

### Removed

- The velocity-flip cushion detector. It found 2 of 23 contacts, and fired on
  a jump shot 25 in from any rail.

### Fixed

- **Positions on broadcast footage**: every ball was placed 2.4 in (near
  rail) to 4.0 in (far rail) too far from the camera. The near-rail bounce on
  `fedor_shot` now turns round 0.2 in off the cushion, where it was 2.1 in
  short.
- **Cushion contacts**: 5 → 49 of 55 found across the synthetic clips, none
  false. `fedor_shot` reports its 4, and `fedor_jump` its 4.
- **Speed error on broadcast-style video**: 8.8% → 2.2% median, and 25 →
  11 in/s at the 95th percentile, measured on the old view. Real-clip speed
  jitter fell from 13.7 / 4.9 / 4.6 to 7.0 / 3.7 / 2.2 in/s.
- **`albin_fedor`**:
  - The cue ball in the pocket jaws is no longer a phantom scratch, and no
    longer renumbered `#10`.
  - The 5-ball's pot into the far-left corner (frame 566) is now detected.
    It had been missed, and the recorded ground truth wrongly called any pot
    on that clip a phantom.
- `run_report.py --out-dir` crashed for a directory outside the repo.

## [2.0.0] — 2026-09-22

Compared against `a2a662a` ("Update README.md", 2024-05-19), the last commit
on GitHub. The 16 commits since then add about 10,000 lines across 35 files.

The v1 script only worked after hand-picking HSV bounds for each ball in each
video. v2 is a new package, `billiards/`, that measures the cloth colour and
table geometry from the video itself and works in table inches from then on.
Nothing needs retuning for a new clip.

### Added

- **`billiards/` package**, replacing the single-file script:
  - `table.py`: measures the cloth colour across 25 sampled frames (circular
    hue mode, MAD-based windows) and calibrates the table.
  - `geometry.py`: fits the four cushions as robust lines and intersects them
    for the corners, so pockets no longer throw the corners off. It works out
    the table's orientation from projective geometry, and gives the
    perspective scale at each point.
  - `detect.py`: detects anything on the bed that is not cloth, sized from the
    homography. It excludes pockets, splits touching balls (distance-transform
    peaks plus radial-symmetry voting on the colour gradient) and takes a Lab
    colour signature for each ball.
  - `kalman.py`: one filter per ball in table inches, with rolling friction
    and adaptive process noise.
  - `track.py`, `assignment.py`: globally optimal assignment on distance plus
    colour, a tentative → confirmed → coasting → deleted lifecycle, and a
    pure-NumPy Hungarian solver for when SciPy is missing.
  - `events.py`: collisions, cushion contacts, pots and balls struck, all
    measured in physical units.
  - `shots.py`: groups events into readable shots, e.g.
    `shot 1: CUE struck -- hit #8 first -- potted #8`.
  - `render.py`, `video.py`: the annotated video, an overhead diagram drawn
    from table coordinates, and CSV/JSON export.
  - `pipeline.py`: runs the stages in order, handles camera cuts and dissolves,
    and skips repeated frames.
  - `config.py`, `cli.py`: configuration in physical units, table presets, and
    the command line.
- **CLI** with `track` (the default), `calibrate` and `dump-config`
  subcommands, and a `billiards` console script.
  - Presets: `pool-9ft`, `pool-8ft`, `pool-7ft`, `snooker-12ft`, `carom-10ft`.
  - Options: `--start`/`--end`, `--table-corners`, `--max-width`, `--debug`,
    `--no-overhead`, `--overhead-inset`, `--quiet`.
  - `calibrate --save-preview` writes a PNG of the detected table and cloth
    mask.
- **`config/default.yaml`**: every value is in inches, seconds or a ratio,
  never pixels.
- **Tools**:
  - `tools/make_synthetic_clip.py`: a physics-simulated break with exact
    ground truth.
  - `tools/evaluate.py`: MOT scoring against that ground truth.
  - `tools/run_report.py`: noise metrics for the real clips. Each run is
    appended to `reports/run-log.json`, and `results/` is regenerated.
- **Tests**: 76 of them in `tests/test_units.py` and `tests/test_pipeline.py`,
  including end-to-end accuracy, camera-cut, repeated-frame and
  optional-dependency fallback tests.
- **CI** (`.github/workflows/tests.yml`): tests on Python 3.9 and 3.12 with
  headless OpenCV, plus an accuracy job that uploads the scored report.
- **Packaging**: `pyproject.toml` (version 2.0.0), `requirements.txt` and
  `requirements-dev.txt`.
- **Docs**: `UPGRADE_NOTES.md`, a rewritten `README.md`, and
  `docs/images/demo_synthetic.png`.

### Changed

- `main.py` is now a small entry point that calls `billiards.cli.main`.
- Contact is tested at the balls' closest approach *during* the frame, not
  only where they end up. The gate is 1.25 ball diameters and requires the gap
  to be shrinking.
- "Cue ball" and "8 ball" are now assigned across the whole table, so there is
  exactly one of each (none of the 8 on carom tables).
- The event `SHOT_START` is renamed to `BALL_STRUCK`.
- The overhead diagram and status lines now sit in a bar below the video
  instead of over the table. The bar also shows the shot in progress.
  `--overhead-inset` restores the old corner layout.
- A track may coast only one frame for each frame it was actually observed, up
  to the previous limit.
- Speed: 9.8 → 23.3 fps on 1280×720 footage, and 27.3 fps on rewrapped
  broadcast clips.
- `.gitignore` now covers the virtualenv, caches and run outputs, including
  `results/`.

### Fixed

- **v1 bugs:**
  - The video writer was sized for the source frames but fed resized ones, so
    the output file was unplayable.
  - Frames were written before anything was drawn on them.
  - The output frame rate was hard-coded to 10.
  - `waitKey(0)` stopped for a key press on every frame.
  - The hard-coded input file `kpc-break.mp4` is not in the repo.
  - Lists grew without limit and were rescanned every frame.
  - `oj_past_center` was used before it was assigned.
  - The table was assumed to be portrait.
  - Corners were thrown off by the pockets.
- **Real broadcast footage:**
  - Wrong table orientation from the standard end-rail camera angle.
  - Balls were sized as flat discs, so they split into phantom halves.
  - Background banners bled into the cloth mask.
  - The black ball was classified as cloth.
  - Balls were detected on the crowd after a camera cut, and during a
    crossfade.
- **Duplicated frames:** 25 fps content rewrapped at 37.7 fps was measured
  twice. Those frames are now replayed with the correct `dt`, which removes
  the speed jitter and false "struck" events they caused.
- **Non-balls:** the player's hand, forearm and cue are no longer reported as
  balls. On `fedor_shot.mp4`, tracks went from 62 to 8.
- **Stripes:** striped balls are no longer labelled "CUE".
- **Event detectors:**
  - Cushion contacts were never detected at break speeds.
  - "Ball struck" fired repeatedly, or not at all. It now uses a hysteresis
    band.
- **Bugs introduced during the rewrite and caught by tests:**
  - A crash on frames with no detections.
  - `np.cross` on 2-D vectors, which NumPy 2.0 removed.
  - Run totals reset after recalibration.

### Moved

- `Indexer.py`, `PoolTable.py` and `ball_hsv_values.txt` → `legacy/`.
- The original `main.py` → `legacy/main_legacy.py`, unchanged.

### Accuracy (synthetic break with ground truth)

| Metric | Value |
|---|---|
| MOTA | 0.936 |
| Precision / recall | 1.000 / 0.937 |
| ID switches | 2 |
| Median position error | 0.150 in |

### Known issues

Carried over from the latest entry in `reports/run-log.json`:

- `fedor_shot`: the cue ball's bounce off the far rail is not reported,
  because the ball is inside the excluded rail margin while it bounces.
- About 4 in/s of speed jitter remains on rewrapped broadcast footage, because
  the true frame spacing is not recorded in the file.
- `albin_fedor`: the cue ball is renumbered at the pocket, and its peak speed
  reads 572 in/s on a 160 in/s shot because of motion blur.
- `albin_fedor`: 3 phantom tracks remain, probably from the cue shaft and the
  player's hand.

### Commits

| Commit | Summary |
|---|---|
| `ba09480` | Rewrite tracking: calibration-first pipeline, no per-video colour tuning |
| `98d2dbc` | Make it work on real broadcast footage |
| `9763f6a` | Speed: 2.4x faster, identical output |
| `320fccf` | Tell a stripe from the cue ball, and document the whole rewrite |
| `0cf8dae` | Add CI, packaging metadata and a regenerated default config |
| `93ff513` | Group events into shots, and fix the threshold bug that hid them |
| `53813bf` | Pin the failure messages, and stop reaching across module boundaries |
| `5f9a29c` | Cover the optional-dependency fallbacks |
| `2f3e575` | Fix two event detectors that were quietly wrong on real footage |
| `6bbba3d` | Assign "cue ball" and "8 ball" across the whole table, not ball by ball |
| `86f750c` | Refresh the demo image |
| `838813e` | Ignore the results/ run-output directory |
| `051f11b` | Stop reporting the player's hand, the cue and a camera cut as balls |
| `e33a39b` | Stop measuring the same frame twice, and find contact during the frame |
| `6c08e6a` | Rewrite results/ on every measured run, not just the log |
| `b46bb6a` | Move the overhead diagram out of the picture |

## v1 — 2024-05-19

The original ENG 301 Computer Vision final project (Fulbright University
Vietnam, Spring 2024). It tracked one cue ball and one object ball using HSV
bounds picked by hand for each video. The code is kept in [`legacy/`](legacy/).
