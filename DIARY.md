# 📓 Development Diary: Billiard Ball Tracking

What state the project was in on each day, one table row per step. The
reasoning and measurements are in [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md), the
list of changes in [`CHANGELOG.md`](CHANGELOG.md).

> `#8` in a shot log is the **tracker's ID** for a ball, not the number on it.

---

## 🧭 Where the project stands

*Last updated: **28 Sep 2026***

| Area | Status | Notes |
|---|:---:|---|
| The app | ✅ | `python main.py app`: says what to do next, one button per video; library, set-up, results, live |
| Live tracking | ✅ | Camera, stream, or a video replayed live; tested on replays only |
| YouTube links | ✅ | Paste a link, pick the minutes, see how long first; tried on one YouTube final |
| Finding the table | ✅ | 11 real venues, 7 cloths, 3 floors, 4 cameras; grey cloth found by itself |
| Broadcast cuts | ⚠️ | Balls keep their ids across cuts and cameras; still 3–4× too many ids on real minutes (Mosconi 44, Premier League 33 for ≤10) |
| Finding the balls | ⚠️ | Balls against the far cushion unseen (an app option finds them, plus a phantom) |
| Ball numbers 1–15 | ✅ | 72–97% right on synthetic; right in all 3 real shot logs |
| Cushion contacts | ✅ | 20 of 23 on synthetic, none false; real clips as in the video |
| Pots | ⚠️ | `fedor_shot`'s 2 found; `albin_fedor`'s 5 missed (far-left corner) |
| Collisions | ⚠️ | 4 of 6 on synthetic, 3 false |
| Ball speeds | ✅ | 2.7% median error (4.9% screen-recorded) |
| Tests | ✅ | 152 pass; none yet for today's cut handling |
| On GitHub | ⚠️ | 28 Sep morning pushed (`1387635`); venue fixes (`cfb3b8b`) and cut handling committed, not pushed |

**Key numbers**: synthetic break (`reports/run-log.json`, 28 Sep 12:53, as on 26 Sep)

| | Constant-rate | Screen-recorded style |
|---|---|---|
| MOTA | 0.844 | 0.816 |
| Precision / recall | 1.000 / 0.846 | 0.994 / 0.822 |
| Speed error | 2.7% | 4.9% |

⚠️ Lower than 23 Sep's 0.92 because the benchmark got harder on 23 Sep evening
(real ball colours on the broadcasts' cloth), not because tracking got worse.

---

## 🗓️ Timeline at a glance

| Day | What happened | State at the end of the day |
|---|---|---|
| **15 May 2024** | v1 starts from Stuart Grieve's table-warp code | Warps still photos of a table to top-down |
| **18–19 May 2024** | v1 finished for ENG 301 | Follows the cue ball on one hand-tuned clip |
| **22 Sep 2026** | v2: full rewrite, 16 commits | Tracks every ball, no tuning, measured accuracy |
| **23 Sep 2026** | Work computer: clock, ball height, events, identity, ball numbers | Positions right on broadcasts; cushions 5 → 40 of 44 |
| **26 Sep 2026** | The app, live mode, robustness matrix | Works in a browser; tested on 14 kinds of footage |
| **27 Sep 2026** | App made easier to follow | One next step per video, a guide at the top |
| **28 Sep 2026** | YouTube links; other venues; balls kept through cuts | Paste a link, track the minutes picked; the table found at 11 venues; a broadcast minute lists 26–44 balls, not 48–129 |

---

## 📅 15 May 2024: v1 begins

**State:** `PoolTable.py` and `Indexer.py` warp a *still photo* of a table to a
top-down view. Written by Stuart Grieve in 2015 for his *PoolTable* project
(per their original headers). No video tracking yet.

## 📅 18–19 May 2024: v1 finished

**State:** follows the cue ball (optionally one object ball) on the clip it was
tuned for. Hand-picked HSV box per ball → biggest blob → joined to the path
within 100 px → "collision" within 20 px → warp top-down.

- ✅ A clean cue-ball path and top-down warp on real broadcast footage.
- ❌ Didn't run as committed (opened a missing `kpc-break.mp4`); colours
  hand-tuned per ball per clip; object tracking off, so collisions never ran;
  `waitKey(0)` paused every frame; the saved video was unplayable.

*(Judged from the code and v1's screenshots, since it can't run as committed.)*

---

## 📅 22 Sep 2026: The big rewrite (v2)

*Other computer, with Claude Code. 16 commits, 04:03–16:36.*

**State at the end of the day:** every ball tracked with no tuning · MOTA 0.936,
precision 1.000 · 23 fps at 720p · tracking pauses on camera cuts · 76 tests.

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | v1 needed hand-tuned colours; nothing measured progress | Measure the cloth, fit the table edges, detect "not cloth", Kalman in inches, physics simulator for ground truth (`ba09480`) | MOTA 0.934, 33 tests |
| 2 | Real footage: table read backwards, balls split in half, banners as cloth, balls in the crowd | Only orientations a real camera can produce; balls sized as spheres; cloth re-measured inside the table; pause on cuts (`98d2dbc`) | MOTA 0.939, 0 ID switches |
| 3 | 9.8 fps | Perspective scale tabulated once; 64-point colour samples (`9763f6a`) | 23.3 fps, same output |
| 4 | Every stripe called "CUE" | A stripe has one strong colour band, the cue ball none (`320fccf`) | Stripes labelled right |
| 5 | Strikes never reported: two thresholds | One threshold; events grouped into shots; CI (`0cf8dae`…`5f9a29c`) | Shot logs appear. ✏️ The "five pots" in the 05:21 commit were phantoms |
| 6 | 0 cushions; one ball "struck" 8× in 0.5 s | Allow for travel since the last frame; a gap between the two "struck" thresholds (`2f3e575`) | Strikes 17 → 4, 19 → 10, 50 → 26 |
| 7 | 2 "cue" and 4 "8" balls on grey cloth | Each role assigned across the whole table (`6bbba3d`) | 1 cue ball, 1 eight ball |
| 8 | "CUE" on the cue stick; 62 tracks for 8 balls | Reject knuckles unless explained by balls (❌ 1st try cost 9 pts of recall); catch dissolves; coast only as long as seen (`051f11b`) | Tracks 62 → 10, 26 → 12, 10 → 9 |
| 9 | Real clips have no ground truth | `tools/run_report.py` scores them against physics; every run logged | `reports/run-log.json` |
| 10 | 1 frame in 3 is a copy: a steady ball read 30, 75, 9, 124 in/s | Skip exact copies; contact tested *during* the frame (❌ copy check made it 21 → 13 fps until moved into OpenCV) (`e33a39b`) | Speed jitter 7.7 → 4.6 in/s, 27 fps |
| 11 | Stale `results/`; the diagram covered the table | `results/` rewritten every run; diagram in a bar below (`6c08e6a`, `b46bb6a`) | ✏️ *23 Sep:* `albin_fedor` has 2 real pots, not 0 |

**Open:** cushion bounces missed · ~4 in/s speed jitter · `albin_fedor`'s cue
ball renumbered at the pocket · 3 phantom tracks on `albin_fedor`.

---

## 📅 23 Sep 2026: Clock, ball height, events, identity

*Work computer: no GitHub (PyPI works), scratch venv. Nothing committed yet.*

### State at the end of the day

| | |
|---|---|
| ✅ Positions | Balls were 2–4 in too far from the camera; fixed |
| ✅ Speeds | Error 8.8% → 2.2% on broadcast-style video |
| ✅ Cushions | 5 → 40 of 44 on synthetic, none false; `fedor_shot` reports its 4 |
| ✅ Identity | `albin_fedor`: no phantom scratch, no `#10` |
| ✅ Tests | 88 pass (76 at the start) |
| ⚠️ Open | Outline includes the clothed cushions · far-cushion balls unseen · collisions 8 of 11, 4 false · one phantom pot by a side pocket (synthetic) |
| ❌ Missing | Ball numbers: the next big feature ✏️ *added that evening, row 13* |

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | GitHub blocked; `.venv` belongs to the other computer | Compare with `origin/main`; scratch venv from PyPI | 76/76 pass; every 22 Sep number reproduced |
| 2 | Watching the videos: `fedor_shot`'s cue ball hits 4 rails, 0 reported; `albin_fedor`'s shows as `#10` | (list of what to fix) | Every ball found; 8 label and the 2-ball pot right |
| 3 | The "unfixable" 4 in/s speed jitter | 🔍 The file is 37.5 fps, but each new frame is just the next 25 fps frame; ~1 step in 7 skips one | Cause found |
| 4 | The simulator couldn't check events or retiming | It records every collision, cushion and pot, and writes screen-recorder-style clips (❌ first MOTA 0.79: our off-by-one in the truth) | Showed only 2 of 23 cushions, 2 of 13 collisions found |
| 5 | The wrong clock | `billiards/clock.py`: moving balls decide 1, 2 or 3 source frames (❌ counting ties drifted to 33 fps; ❌ falling back to the file's clock was worse) | Speed error 8.8% → 2.2%; jitter 13.7/4.9/4.6 → 7.8/3.6/2.2 in/s |
| 6 | An `albin_fedor` pot logged as a phantom | 🔍 The purple 5 really drops at frame 566 | Truth: 2 pots, 1 detectable. A missed pot found |
| 7 | The filter's velocity turns round 4–6 in off the cushion | Cushions from the raw path: toward a rail, then away (❌ corner fit: fine on synthetic, useless on real) | 49 of 55 on 3 synthetic clips, none false |
| 8 | Cue ball turned round 2.1 in short of the near cushion | 🔍 A ball's centre is 1.125 in up. Camera recovered from the table; balls placed at centre height; simulator given a real pinhole camera (❌ wider search: 30 tracks for 7 balls, reverted) | 0.2 in off the cushion; MOTA −0.43 without the fix, 0.92 with |
| 9 | A static rack seen from an end rail hides itself | Truth records visibility; balls < ½ visible not scored | MOTA 0.92–0.93; test bars re-set |
| 10 | `albin_fedor`'s cue ball in the jaws: a scratch, back as `#10` | Dead tracks wait 1.5 s to be revived; roles survive shadows (❌ bouncing hidden balls lost a real pot until pocket-bound balls were exempt) | Every real shot log matches the video; jitter 7.0/3.7/2.2 in/s |
| 11 | Docs behind the code | `UPGRADE_NOTES.md` §11, README, demo image, `CHANGELOG.md`, `legacy/README.md`, `CLAUDE.md`, this diary | Official run logged 13:16 |
| 12 | This diary was getting long (308 lines) | One table row per step; `CLAUDE.md` rule updated to match | 308 → 159 lines |
| 13 | *Evening, written up 26 Sep from the session log:* ball numbers | `billiards/balls.py`: numbers from colour, one assignment for the whole table; simulator gets the broadcasts' colours; cushion-nose fit started | Shot logs say "potted the 2"; ⚠️ left undocumented, 2 tests failing |

---

## 📅 26 Sep 2026: Loose ends, then an app

*Work computer, scratch venv. The machine slept 05:17–11:45 and 13:11–16:46. Committed and pushed at the end.*

### State at the end of the day

| | |
|---|---|
| ✅ App | `python main.py app`: library, set-up, runs, results, live; screenshot in the README |
| ✅ Other footage | 6 cloths, 4 cameras, 480p–1080p, 60 fps, screen-rec: MOTA 0.57–0.88 |
| ✅ Cushions | No phantoms: `fedor_shot` 4, `fedor_jump` 4, as in the video |
| ✅ Tests | 131 pass |
| ⚠️ Open | `albin_fedor`'s 5-ball pot missed · real clips' source rate uncertain · grey cloth needs corners · corner camera 0.57 |
| ✅ Measured | `run_report.py` 17:21, `robustness.py` 17:24; `results/` rewritten; README at these numbers |
| ✅ On GitHub | 22–26 Sep pushed (`0fe9fc9`) |

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | 23 Sep evening work undocumented; 2 tests failing | Read the session log; baseline run logged (04:06) | Harder synthetic benchmark: MOTA 0.844 / 0.816 |
| 2 | Cushions 24 and 39 in from any rail | An approach expires once the ball is seen standing (❌ plain age limit → lost a bounce hidden in the jaws) | `fedor_shot` 5 → 4, `fedor_jump` 5 → 4; synthetic unchanged |
| 3 | Balls against the far cushion unseen | ❌ search past the far edge: recall +1.6 pts, but `albin_fedor`'s 8 splits into the nose shadow (8 tracks for 7) → kept as an off switch | No change by default |
| 4 | Clock read 57.9 fps off 25 fps, 41.1 off a 37.5 fps file | Rate bounded by frame counts; velocities follow its changes; 5% snap (❌ fitting a period to measured steps: time scale unobservable) | 25.0 on the test clip; long synthetic speed error 5.2 → 4.9% |
| 5 | Test bars from before the harder benchmark | Re-set to measured; 3 tests; `default.yaml` regenerated | 115 pass |
| 6 | Everything was command line | `billiards app`: library, set-up with draggable corners, background runs, results with a synced top-down view, timeline, speed chart | 13 app tests; checked in headless Edge |
| 7 | Browsers can't play `mp4v` | Writer probed: H.264 via `imageio-ffmpeg`, else OpenCV `avc1`, else VP8 (❌ Windows' own H.264: ~50 Mbit/s) | 375 frames → 427 KB |
| 8 | No live mode | Camera / stream / video replayed live; newest frame only; recording saved as a run | `fedor_shot` live: 38 fps in, ~27 tracked, same shot, 3 of 4 cushions |
| 9 | Only one venue and camera ever tested | `tools/robustness.py`: 14 variants vs ground truth (❌ 3 cameras read −0.4…−0.75 → scoring ignored the table's symmetry) | Cloths 0.74–0.87; side 0.72, ceiling 0.88, corner 0.57 |
| 10 | Grey cloth: a banner taken for the table | Tiny "tables" refused; hand-placed corners measure the cloth inside; grey modelled without hue | Grey + corners: 0 → 0.79 |
| 11 | 1080p −0.04, then 480p −0.03 | Simulator lines scale with the picture; nose search 6 → 10 in | 1080p 0.79, 480p 0.77 |
| 12 | `albin_fedor`'s 5 pot | 🔍 13:04's "potted the 5" was a stray track by a side pocket at frame 750 (❌ far-cushion search finds the real one, frame 568, but adds a phantom → per-video app option) | ⚠️ Now reported missed |
| 13 | Docs | README app section, `UPGRADE_NOTES.md` §12, CHANGELOG | ⏳ Final run interrupted by a laptop restart |
| 14 | The restart killed the final runs | Re-ran `run_report.py --ground-truth` (17:21), `robustness.py` (17:24) | Synthetic 0.844 / 0.816, as at 13:04; 14 variants as rows 9–11; `albin_fedor` clock 30.0 → 31.1 fps |
| 15 | Grey + corners 0.79 was a one-off | Re-run with the true corners placed | ✅ 0.79 |
| 16 | App opened empty; README screenshot missing | 3 sample clips tracked in the app; captured over CDP (❌ headless `--screenshot`: video stuck on frame 0, or blank) | `docs/images/app_results.png` |
| 17 | README at 23 Sep numbers, "88 tests"; `default.yaml` still 6 in | README accuracy + other footage + limitations; `UPGRADE_NOTES.md` §1; `dump-config` | Docs match the 17:21 run |
| 18 | 22–26 Sep work only on this laptop | 23 + 26 Sep as one commit, `.idea/` left out; push worked (GitHub reachable after all) | 17 commits on GitHub |

---

## 📅 27 Sep 2026: An app that says what to do

*Work computer, scratch venv. Page changes only: the tracker is untouched, so `results/` still matches it. Committed and pushed.*

### State at the end of the day

| | |
|---|---|
| ✅ App | The library says what to do next; one blue button per video |
| ✅ Tests | 131 pass |
| ⚠️ Open | Not yet tried by the user after the change · 26 Sep's open tracking issues unchanged |
| ✅ On GitHub | Pushed the same day |

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | "The app is very hard to use": unclear where to start and what to click | 🔍 Each video had 4 equal buttons; the blue one, **Track**, re-ran a tracked video; the picture opened set-up | The user's 18:55 re-run of `albin_fedor` was exactly that |
| 2 | No starting point | Library guide ① add a video → ② track it → ③ see the results, current step lit, one sentence on what to do | Empty, untracked, tracking and done states checked at 1280×650 |
| 3 | 4 equal buttons per video | One blue button for the next step (Track → Watch progress → See results); the picture does the same; the rest are small links | Clicked through Track → progress → results on a fresh workspace |
| 4 | "Upload" beside "Add footage"; "Runs"; 6 buttons over the results | One **Add videos** dialog; Runs → **Results**; one **Download** menu; set-up and results pages say what to do | README text and screenshot updated |
| 5 | Live showed "null" and a broken picture before starting | A missing name was passed to the page as `null`; the empty picture is hidden | Fixed; no JavaScript errors on any page |

---

## 📅 28 Sep 2026: A YouTube link, other venues, and cuts

*Work computer, scratch venv (+ yt-dlp). Rows 1–6 pushed (`1387635`); rows 7–11 committed as `cfb3b8b`, rows 12–18 in the commit after it, not pushed. `results/` regenerated at 12:53 (row 18).*

### State at the end of the day

| | |
|---|---|
| ✅ Links | Paste a link, pick the part, see the estimate; only that part is downloaded, then tracked (app and command line) |
| ✅ Estimate | The example final: 1 h 45 min for all 61 min in the app; a 3–5 min part 5–8 min |
| ✅ Other venues | The table found at all 11 tried; grey cloth by itself |
| ✅ Cuts | A ball keeps its id across a cut and a change of camera (simulated cuts: MOTA 0.40 → 0.78) |
| ⚠️ Broadcasts | Still 3–4× too many ids a minute (Mosconi 44, Premier League 33, UK Open 26); replays tracked as play |
| ⚠️ Not tried | Sites other than YouTube; the other computer (needs yt-dlp, and Node or Deno); real CCTV footage |
| ✅ Tests | 152 pass (131 at the start) |

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | "Give it a YouTube link"; the example is 61 min at 60 fps | 🔍 Download 20:00–24:00 and track it (❌ part of the direct file: 403 → streamed copy) | 4 min in 17 s; 32 cuts recovered, 23 shots |
| 2 | Pick a few minutes, and say how long first | `fetch.py` + app downloads: link box in *Add videos*, part, game from the title, *Download and track* | 20:00–21:00 downloaded and tracked in the app in 84 s |
| 3 | The estimate | ❌ median of recent runs → one fast minute made the hour 1 h 25 min → fastest-to-slowest range, broadcast pace kept until 3 runs | "About 7 to 8 min" for 5 min |
| 4 | The first hour figure, 2 h 50 min | 🔍 That was `track -o`, which draws the diagram: app 35.4 fps, `-o` 21.5, `--no-overhead` 29.2 on the same 4 min | ✏️ The hour is 1 h 45 min in the app |
| 5 | The command line | `billiards <link> --start 20:00 --end 25:00`; `--numbers`; m:ss times | 30:00–30:20 fetched, 10-ball numbers, tracked |
| 6 | "6 min 60 s" on the page | Round before splitting minutes | Fixed; 18 new tests, no network needed |
| 7 | "Couldn't track anything" (US Open); "not very robust" at another venue | 🔍 Its royal-blue floor was taken for the blue-grey cloth; the simulator's floor was always grey | Reproduced: simulated blue floor 0.81 → 0 |
| 8 | The cloth taken from the commonest hue | Try 7 colours and greys, keep the most table-like outline (❌ grey on grey took the whole picture → an outline off the picture is never a table) | Your US Open 3 min: 0 → 17 shots; simulated grey, blue floor, red floor 0 → 0.79 / 0.81 / 0.86 |
| 9 | UK Open: 3 cameras, no agreeing outline | Calibrate on the biggest agreeing group of frames, not the median of all | Table found; sample clips unchanged |
| 10 | UK Open: score-bar icons tracked; 95 contacts during a zoom | Never adopt an outline off the picture; pause when a zoom runs the cloth off it | 96 → 15 contacts; `run_report` + robustness unchanged elsewhere |
| 11 | No real footage from other venues | `tools/venues.py`: a minute each from 11 venues (Mosconi, UK Open, heyball, snooker…) | All 11 found; phantoms remain (`results/venues.png`) |
| 12 | "Reliably track tournament videos": 91 / 129 / 48 balls in a minute (≤10) | 🔍 Tracker rebuilt at every changed outline (every 5 s on grey cloth, no cut needed); ids restarted at 1, so the CSV mixed balls | Cause found |
| 13 | Every cut forgot the balls | One tracker; at a cut balls are set aside, rolled on, re-found by position or colour (❌ last-seen position: balls rolling at the cut not found) | Simulated cuts: ids per ball 2.3 → 1.2 |
| 14 | Another camera put the balls elsewhere | Views remembered; a new view turned or mirrored to match the balls (❌ turning only: the across-the-table camera is a mirror image) | That segment: MOTA −0.69 → 0.70 |
| 15 | Table stale when the camera pushes in | Re-check fixed (never ran at 60 fps); "view still fits" test; camera followed by feature matching (❌ refitted outlines: a player over a rail read as a move) | Premier League: 8 moves followed |
| 16 | New views 6 in too big; one "table" crossed itself | Half-size mask keeps the nose line; corner orders aligned; outlines must follow the cloth | 6.25 → 0.8 in |
| 17 | No measure of cuts | `robustness.py` *cuts* variant; `evaluate.py` ids per ball | MOTA 0.404 → 0.782, precision 1.000 |
| 18 | Checks | 152 tests; `run_report` 12:53 and 16 robustness variants unchanged; venue minutes | Balls 91 → 44, 129 → 33, 48 → 26; ✏️ Mosconi was 35 before its overhead camera was tracked |

---

## ✍️ How to add a day

Copy this to the bottom, then update **Where the project stands** and the
**Timeline**. One row per step, one short phrase per cell; a day should fit on
one screen. Detail goes in `UPGRADE_NOTES.md` and commit messages.

```markdown
## 📅 <date>: <short title>

*<which computer / anything unusual>*

### State at the end of the day

| | |
|---|---|
| ✅ Works | ... |
| ⚠️ Open / ❌ Missing | ... |

| # | 🧩 Problem | 🔧 Fix | 📈 Result |
|---|---|---|---|
| 1 | what looked wrong | what fixed it (❌ tried X → why it failed) (`<commit>`) | before → after |
```

Numbers come from `tools/run_report.py` / `reports/run-log.json` or
`tools/evaluate.py`; say when something wasn't run. If later work shows an entry
was wrong, add a ✏️ *Correction* in that row rather than rewriting it.
