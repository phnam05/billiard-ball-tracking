"""Configuration for the billiard tracking pipeline.

Design rule for this file: **every tunable is expressed in physical units**
(inches, inches/second, seconds) or as a dimensionless ratio -- never in pixels.

The old pipeline hard-coded pixel thresholds (``distance < 20``, ``< 100``,
``sense = 10`` around hand-picked HSV bounds).  Those numbers are only valid for
one video at one resolution with one camera placement, which is why retuning was
needed for every clip.  Physical units are resolution independent: once the table
homography is known, the pipeline converts inches to pixels itself.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:  # PyYAML is optional; JSON always works.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    yaml = None  # type: ignore


# --------------------------------------------------------------------------
# Table specifications.  Dimensions are of the *playing surface* (bed), i.e.
# cushion nose to cushion nose, which is what the detected cloth polygon spans.
# --------------------------------------------------------------------------

TABLE_PRESETS: Dict[str, Dict[str, float]] = {
    # American pool
    "pool-9ft": {"length_in": 100.0, "width_in": 50.0, "ball_diameter_in": 2.25},
    "pool-8ft": {"length_in": 88.0, "width_in": 44.0, "ball_diameter_in": 2.25},
    "pool-7ft": {"length_in": 78.0, "width_in": 39.0, "ball_diameter_in": 2.25},
    # Carom / three-cushion (no pockets)
    "carom-10ft": {"length_in": 111.8, "width_in": 55.9, "ball_diameter_in": 2.42},
    # Snooker
    "snooker-12ft": {"length_in": 140.5, "width_in": 70.0, "ball_diameter_in": 2.07},
}
DEFAULT_PRESET = "pool-9ft"

_SECTION_TYPES: Dict[str, type] = {}


@dataclass
class TableConfig:
    """Physical table description plus how hard to work at finding it."""

    preset: str = DEFAULT_PRESET
    #: Filled in from ``preset`` at load time unless explicitly supplied.
    length_in: float = 100.0
    width_in: float = 50.0
    ball_diameter_in: float = 2.25

    #: Resolution of the rectified (overhead) table image, in pixels per inch.
    #: 8 px/in gives an 800x400 canvas for a 9ft table -- plenty for tracking
    #: and cheap to warp.
    overhead_px_per_inch: float = 8.0

    #: Number of frames sampled across the clip to estimate the cloth colour
    #: and the table corners.  Sampling many frames and taking a robust median
    #: is what makes calibration survive a player leaning over the rail.
    calibration_frames: int = 25

    #: How far a corner may drift before we decide the camera actually cut or
    #: panned and a recalibration is needed.  Fraction of the table short side.
    recalibration_tolerance: float = 0.06

    #: Re-check the table geometry every N frames (0 disables).
    recalibration_interval: int = 150

    #: Shrink the detected table polygon by this many ball diameters before
    #: looking for balls, so cushions/rails/pocket jaws do not create blobs.
    bed_margin_ball_diameters: float = 0.35

    @property
    def ball_radius_in(self) -> float:
        return self.ball_diameter_in / 2.0

    @property
    def overhead_size_px(self) -> Tuple[int, int]:
        """(width, height) of the rectified table image, long axis horizontal."""
        return (
            int(round(self.length_in * self.overhead_px_per_inch)),
            int(round(self.width_in * self.overhead_px_per_inch)),
        )


@dataclass
class ClothConfig:
    """Automatic cloth-colour estimation.

    There is deliberately no ``lower_bound``/``upper_bound`` here.  The cloth
    colour is measured from the video itself; these knobs only control *how
    tolerant* that measurement is, and the defaults work for green, blue and
    red cloth under both TV lighting and a phone camera.
    """

    #: Hue/saturation are stable under lighting change; value is not.  We gate
    #: tightly on H/S and loosely on V.  Widths are in units of robust standard
    #: deviations (MAD-derived) around the measured mode.
    hue_sigmas: float = 4.0
    sat_sigmas: float = 4.5
    val_sigmas: float = 6.0

    #: Absolute floors, so a perfectly uniform cloth still gets a usable window.
    min_hue_halfwidth: float = 6.0
    min_sat_halfwidth: float = 40.0
    min_val_halfwidth: float = 70.0

    #: The value window is deliberately asymmetric.  A shadow can only make the
    #: cloth *darker*, never brighter, and the shadow a ball casts is what welds
    #: neighbouring balls into one blob if it is treated as foreground.  Hue and
    #: saturation stay tight, so widening downwards costs almost nothing: a dark
    #: ball is excluded by its hue or its low saturation, not by its brightness.
    shadow_value_factor: float = 2.2

    #: Pixels darker / less saturated than this are never cloth.  Kills shadow
    #: under the rail and the black bars around letterboxed video.
    min_value: int = 25
    min_saturation: int = 25

    #: Downscale factor used when histogramming frames, for speed.
    analysis_scale: float = 0.5


@dataclass
class DetectorConfig:
    """Ball detection.

    Every size gate is relative to the *expected* ball area at that image
    location, which the pipeline derives from the homography.  Nothing here is
    in pixels, so the same numbers work for a 480p phone clip and a 4K
    broadcast.
    """

    #: A blob must be within [min, max] x the expected single-ball area to be
    #: considered.  The generous upper bound lets clusters through; they are
    #: then split by watershed rather than discarded.
    min_area_ratio: float = 0.22
    #: Generous on purpose: a full rack of 15 balls plus their shadows is a
    #: single ~20x blob, and rejecting it outright (the old ceiling of 9 did
    #: exactly that) means the entire rack is invisible until it breaks apart.
    max_area_ratio: float = 32.0

    #: Above this ratio the blob is assumed to be several touching balls and is
    #: split with a distance-transform watershed.
    split_area_ratio: float = 1.55

    #: 4*pi*A/P^2.  A perfect disc is 1.0.  Applied only to blobs that were not
    #: split (split fragments are allowed to be scruffy).
    min_circularity: float = 0.55

    #: area / convex-hull area.  Rejects ring-shaped and forked blobs.
    min_solidity: float = 0.80

    #: Reject long thin blobs.  This is what removes the cue stick, the bridge
    #: hand and rail highlights, with no colour tuning at all.
    max_aspect_ratio: float = 2.6

    #: Morphological opening radius, in ball radii.
    open_radius_ball_radii: float = 0.22

    #: Distance-transform peak separation when splitting clusters, in ball
    #: radii.
    peak_separation_ball_radii: float = 1.35

    #: Ignore anything closer than this to the bed edge, in ball diameters.
    edge_margin_ball_diameters: float = 0.0

    #: Blank out a disc of this radius (in ball diameters) at each pocket.
    #: A pocket is a dark, round, ball-sized hole sitting permanently on the
    #: bed, so without this it is detected as a stationary ball on every single
    #: frame -- four phantom balls on a pool table.  0 disables.
    pocket_exclusion_ball_diameters: float = 1.45

    #: When splitting a cluster, a peak of the distance transform is only a ball
    #: if its height is close to the ball radius.  A wider object (an arm, a
    #: sleeve, a pile of chalk) has a much taller ridge, so this band is what
    #: keeps the generous max_area_ratio above from admitting junk.
    split_peak_min_ratio: float = 0.55
    split_peak_max_ratio: float = 1.75

    #: Cap on detections per frame (guards against a pathological frame).
    max_detections: int = 40


@dataclass
class TrackerConfig:
    """Kalman + Hungarian multi-object tracker.

    Runs entirely in table coordinates (inches), where ball motion really is
    close to constant-velocity-with-friction.  The filter's model therefore
    matches reality and every gate below is physically meaningful.
    """

    #: Fastest plausible ball.  A hard break is ~30 mph = 528 in/s; allow
    #: headroom.  Used to derive the association gate: a detection further than
    #: max_speed*dt from a prediction cannot be the same ball.
    max_speed_in_s: float = 700.0

    #: Extra slack on the gate, in ball diameters, to absorb detection jitter
    #: at low speed.
    gate_padding_ball_diameters: float = 1.2

    #: Rolling friction: velocity decays with this time constant, in seconds.
    velocity_tau_s: float = 3.6

    #: Base process noise, as an acceleration standard deviation in in/s^2.
    #: This is the *quiet* value: a freely rolling ball only accelerates through
    #: friction (a few in/s^2) and the sliding phase right after a hit (~80).
    #: Collisions and cushion bounces are handled by raising this adaptively --
    #: see BallKalman._process_noise -- rather than by keeping it permanently
    #: high, which would make a stationary ball read as moving at 15-20 in/s.
    accel_std_in_s2: float = 60.0

    #: Ceiling on the adaptive process-noise multiplier.
    manoeuvre_gain_max: float = 25.0

    #: Measurement noise standard deviation, in inches.
    meas_std_in: float = 0.22

    #: Initial velocity uncertainty, in in/s.
    init_vel_std_in_s: float = 120.0

    #: Weight of colour dissimilarity in the association cost, relative to
    #: normalised spatial distance.  Colour is what keeps IDs correct through a
    #: collision, where spatial cues alone are ambiguous.
    color_cost_weight: float = 0.6

    #: Detections whose colour distance exceeds this are never matched to the
    #: track, however close they are.
    max_color_distance: float = 42.0

    #: Track lifecycle, in frames.
    min_hits_to_confirm: int = 3
    max_age_coasting: int = 45
    max_age_tentative: int = 3

    #: A ball is treated as stationary below this speed, which suppresses
    #: trajectory jitter when nothing is moving.
    stationary_speed_in_s: float = 4.0


@dataclass
class EventConfig:
    """Collision / cushion / pot detection, all in physical units."""

    #: Ball-ball contact when centre distance < this many ball diameters.
    contact_distance_ball_diameters: float = 1.12

    #: Minimum closing speed for a contact to count as a real collision rather
    #: than two balls resting against each other.
    min_closing_speed_in_s: float = 6.0

    #: Minimum change in a ball's velocity vector for a cushion bounce, in in/s.
    min_bounce_speed_change_in_s: float = 18.0

    #: A ball within this many ball radii of a cushion is "at" the cushion.
    cushion_proximity_ball_radii: float = 1.25

    #: A track that vanishes within this many ball diameters of a pocket centre
    #: is reported as potted rather than lost.
    pocket_radius_ball_diameters: float = 1.9

    #: Suppress duplicate events for the same pair within this many seconds.
    refractory_s: float = 0.18


@dataclass
class RenderConfig:
    draw_trajectories: bool = True
    draw_ids: bool = True
    draw_detections: bool = False
    draw_table_outline: bool = True
    draw_hud: bool = True
    draw_events: bool = True
    #: Trajectory history length in seconds (0 = unlimited).
    trail_seconds: float = 4.0
    trail_thickness: int = 2
    #: Render the synthetic overhead diagram alongside the camera view.
    overhead_panel: bool = True
    #: Fraction of the output width the overhead panel occupies.
    overhead_panel_scale: float = 0.34
    font_scale: float = 0.45


@dataclass
class Config:
    table: TableConfig = field(default_factory=TableConfig)
    cloth: ClothConfig = field(default_factory=ClothConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    events: EventConfig = field(default_factory=EventConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    #: Resize the input so the long edge is at most this many pixels.  Detection
    #: quality is scale invariant thanks to the homography, so downscaling is a
    #: pure speed win.  0 disables resizing.
    max_frame_width: int = 1280

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- serialisation -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            section_cls = _SECTION_TYPES.get(f.name)
            if section_cls is not None:
                known = {sf.name for sf in dataclasses.fields(section_cls)}
                unknown = set(value or {}) - known
                if unknown:
                    raise ValueError(
                        "Unknown key(s) {} in config section {!r}".format(
                            sorted(unknown), f.name
                        )
                    )
                kwargs[f.name] = section_cls(**(value or {}))
            else:
                kwargs[f.name] = value
        if data:
            raise ValueError("Unknown top-level config key(s): {}".format(sorted(data)))
        cfg = cls(**kwargs)
        cfg.apply_preset()
        return cfg

    def apply_preset(self) -> "Config":
        """Fill table dimensions from the named preset.

        Explicit values in the config file win; anything left at the dataclass
        default is taken from the preset.
        """
        preset = TABLE_PRESETS.get(self.table.preset)
        if preset is None:
            raise ValueError(
                "Unknown table preset {!r}. Known presets: {}".format(
                    self.table.preset, sorted(TABLE_PRESETS)
                )
            )
        defaults = TableConfig()
        for key, value in preset.items():
            if getattr(self.table, key) == getattr(defaults, key):
                setattr(self.table, key, value)
        return self

    @classmethod
    def load(cls, path: Optional[Union[str, Path]]) -> "Config":
        if path is None:
            return cls().apply_preset()
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to read .yaml config files; "
                    "install it or use a .json config"
                )
            data = yaml.safe_load(text)
        else:
            import json

            data = json.loads(text)
        return cls.from_dict(data)

    def dump(self, path: Union[str, Path]) -> None:
        p = Path(path)
        data = self.to_dict()
        if p.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to write .yaml config files")
            p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            import json

            p.write_text(json.dumps(data, indent=2), encoding="utf-8")


_SECTION_TYPES.update(
    {
        "table": TableConfig,
        "cloth": ClothConfig,
        "detector": DetectorConfig,
        "tracker": TrackerConfig,
        "events": EventConfig,
        "render": RenderConfig,
    }
)
