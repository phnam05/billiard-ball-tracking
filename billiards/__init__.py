"""Billiard ball tracking.

A calibration-first, tracking-based pipeline for following every ball on a pool
table through a video.

Quick start::

    from billiards import Config, RunOptions, run

    summary = run(Config(), RunOptions(video="clip.mp4", output="out.mp4"))

See ``UPGRADE_NOTES.md`` for what changed relative to the original
HSV-threshold script and why.
"""

from .config import TABLE_PRESETS, Config
from .detect import BallDetector, ColorSignature, Detection
from .events import Event, EventDetector, EventType
from .geometry import TableModel
from .kalman import BallKalman
from .pipeline import FrameResult, RunOptions, TrackingPipeline, build_pipeline, run
from .table import CalibrationResult, ClothModel, calibrate, estimate_cloth_color
from .track import MultiObjectTracker, Track, TrackState

__version__ = "2.0.0"

__all__ = [
    "BallDetector",
    "BallKalman",
    "CalibrationResult",
    "ClothModel",
    "ColorSignature",
    "Config",
    "Detection",
    "Event",
    "EventDetector",
    "EventType",
    "FrameResult",
    "MultiObjectTracker",
    "RunOptions",
    "TABLE_PRESETS",
    "TableModel",
    "Track",
    "TrackState",
    "TrackingPipeline",
    "build_pipeline",
    "calibrate",
    "estimate_cloth_color",
    "run",
    "__version__",
]
