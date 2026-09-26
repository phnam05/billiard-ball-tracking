"""A local web app around the tracker: a footage library, set-up, runs, results, live.

Start it with ``billiards app`` (or ``python main.py app``); it opens in the
browser.  See ``billiards.app.server`` for what it serves.
"""

from .server import make_server, serve

__all__ = ["make_server", "serve"]
