# The ball model's real training data

* **`real.csv`** is what training reads: one row per crop, `(clip, file, width,
  frame, x, y, r, label, track)`, cut again from the clip by
  `tools/train_ballnet.py`. The clips are the cached venue minutes and
  CCTV-style clips in `.cache/` and the three sample clips (list:
  `REAL_CLIPS` in `tools/ballnet_data.py`). Labels are `no`, `cue`, `ball`
  (a ball whose colour could not be told), a colour family (`red`, `maroon`
  for brown…), or a family with `-stripe`.
* **`labels/<clip>.json`** are the labels as they were given, by eye from
  contact sheets of each track's crops, 28 Sep 2026: by track id, or by frame
  range where a track jumped between balls. The track ids are those of the
  harvest run of that evening (`tools/ballnet_data.py harvest`, with the
  model off, `BILLIARDS_BALLNET=off`); `tools/ballnet_data.py rows` turned
  them into `real.csv`. A harvest with other code numbers the tracks
  differently, so to add a clip, harvest it, make its sheets, label it, and
  run `rows` for the new clip's file with the old rows kept.

Never add the answer keys' clips (`tools/truth/`) or the `us-open` venue
minute (the same match as the US Open key): they are the test.
