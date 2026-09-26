# legacy/: the original v1 code

These files are the project as it was on GitHub until September 2026. They are
kept for comparison and are not used by the `billiards` package.

| File | What it is |
|---|---|
| `main_legacy.py` | v1's `main.py`, unchanged: the whole tracker, with hand-picked HSV bounds for each ball |
| `PoolTable.py` | Table helpers: cloth colour from histograms, the table outline, and the warp to an overhead view |
| `Indexer.py` | `get_index_of_max` / `get_index_of_min` |
| `ball_hsv_values.txt` | The hand-picked HSV bounds for the cue, blue and pink balls |

## Credit

`PoolTable.py` and `Indexer.py` were not written for this project. They come
from **Stuart Grieve's *PoolTable* project (2015)**, *"Python code to identify
pool balls on a table from screenshots of matches"*. Both files carried his
author headers ("Created on Sun Jul 12 2015, @author: Stuart Grieve"), as the
repository's first upload on 15 May 2024 shows.

When the files were adapted for v1 on 18 May 2024, those headers were removed.
They were restored on 23 Sep 2026. The perspective-warp code in
`TransformToOverhead` is credited, in its own docstring, to a pyimagesearch
tutorial (2014).

The licence of the original *PoolTable* project was not recorded here. Check it
and link the original repository before redistributing these two files.
