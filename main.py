#!/usr/bin/env python
"""Entry point for the billiard ball tracker.

    python main.py                      # open the app in a browser
    python main.py clip.mp4 --show
    python main.py clip.mp4 -o out.mp4 --csv tracks.csv --json run.json
    python main.py calibrate clip.mp4 --save-preview calib.png

Run ``python main.py --help`` for everything else.

The previous version of this file was the entire program: input filename,
output filename, HSV bounds and every threshold were literals in the source,
so each new video meant editing the code.  The implementation now lives in the
``billiards`` package; the original script is preserved in
``legacy/main_legacy.py`` for comparison.
"""

import sys

from billiards.cli import main

if __name__ == "__main__":
    sys.exit(main())
