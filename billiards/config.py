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
    #: A table found automatically on which a ball would be smaller than this
    #: (radius, pixels, mid-table) is taken to be something else of the
    #: cloth's colour -- a banner, a sign -- and calibration fails instead.
    min_ball_radius_px: float = 3.0
    #: Choosing the cloth (``table.choose_cloth``): a frame's outline of a
    #: candidate colour agrees with the others if no corner is further than
    #: this fraction of the picture's diagonal from the median outline.
    candidate_corner_tolerance: float = 0.04

    #: How far a corner may drift before we decide the camera actually cut or
    #: panned and a recalibration is needed.  Fraction of the table short side.
    recalibration_tolerance: float = 0.06

    #: Re-check the table geometry every N measured frames (0 disables).  A
    #: check costs one outline fit, about 17 ms at 720p, so every 30 frames is
    #: half a millisecond a frame; it was 150 until 28 Sep 2026, too slow to
    #: follow a broadcast camera pushing in.  Sooner when the bed stops
    #: looking like cloth (see ``TrackingPipeline._check_view``).
    recalibration_interval: int = 30

    #: Broadcast footage cuts between angles and to replays, and a phone on a
    #: tripod gets nudged.  When that happens the homography is stale and every
    #: physical threshold derived from it is meaningless, so the tracker must
    #: notice rather than keep reporting balls in a crowd.
    #:
    #: The signal used is the one already computed each frame: how much of the
    #: calibrated bed polygon is still cloth-coloured.  On the right shot it sits
    #: near 1.0; after a cut it collapses.  Below this fraction of the
    #: calibrated coverage, for this many consecutive frames, the view is
    #: treated as changed and tracking pauses until the table is found again.
    view_change_coverage_ratio: float = 0.55
    view_change_patience: int = 3

    #: Coverage alone is a lagging signal, because a broadcast dissolve mixes
    #: two shots of the *same* sport: the incoming angle is mostly cloth too,
    #: so the outgoing bed polygon stays cloth-coloured well into the
    #: transition.  On one clip that bought twelve frames in which the crowd,
    #: the rails and a second table were all inside the stale bed polygon, and
    #: those twelve frames spawned thirty phantom balls.
    #:
    #: So the bed is also watched for wholesale change frame to frame.  Balls
    #: and players repaint a small part of it -- the busiest frame of a break
    #: moves about 3% of the bed -- while a cut or a dissolve repaints most of
    #: it at once, measured at 16-32%.  Above this fraction the view has
    #: changed, with no patience: a cut is not ambiguous.
    #:
    #: ``view_change_level`` is how much a pixel must move to count as changed,
    #: as a fraction of full scale, and sits well above sensor and compression
    #: noise.  Set ``view_change_area_ratio`` to 0 to disable the test.
    view_change_level: float = 0.031
    view_change_area_ratio: float = 0.12

    #: The same comparison answers a second question at the opposite end of the
    #: scale: did the bed change *at all*?
    #:
    #: Broadcast clips are routinely 25 fps content rewrapped at 37.7 fps, so
    #: one frame in three is a copy of the one before it.  A copy is not a new
    #: measurement, and treating it as one is double counting: the filter is
    #: told the ball did not move over 1/37.7 s, so its velocity estimate
    #: collapses, and the next real frame then hands it one and a half frames
    #: of travel at once.  On ``fedor_shot.mp4`` that made a cue ball rolling
    #: smoothly at 107 in/s report 30, 75, 9, 124 and 160 in/s on successive
    #: frames; it drove the estimate below the at-rest threshold in the middle
    #: of the roll, which fired a second "struck" event for a ball that had
    #: never stopped; and it stepped the ball over the contact window, so the
    #: shot that potted a ball reported no collision at all.
    #:
    #: A frame that moved less than this many ball areas of the bed is
    #: therefore not measured again.  It is replayed, and the next frame that
    #: *is* measured gets the whole interval as its ``dt``, which is the point:
    #: the geometry was never wrong, only the clock.
    #:
    #: The level is deliberately high -- an eighth of full scale, measured as
    #: the largest step across the colour channels -- because a re-encoded copy
    #: is not bit-identical.  Measured over the three sample clips, a copy
    #: moves *no* bed pixel that far, so the budget below only has to tolerate
    #: the odd compression artefact: it is two hundredths of a ball, about nine
    #: pixels at 720p, against the fifty or more that one ball shifting by a
    #: single pixel repaints.
    repeat_frame_level: float = 0.125
    repeat_frame_ball_areas: float = 0.02

    #: ...but the tracker must not be asked to bridge an unbounded gap.  The
    #: bed is genuinely still between shots -- up to two seconds on the sample
    #: clips -- and extrapolating a Kalman filter across that in one step makes
    #: its association gate wider than the table.  After this many frames in a
    #: row have been dropped, the next one is measured whatever it looks like.
    #: 0 measures every frame, however little changed.
    repeat_frame_max_run: int = 4

    #: Skipping the copies fixes the measurements, not the clock.  The frames
    #: that remain are one *source* frame apart -- 40 ms, in the sample clips --
    #: however many slots of the file separate them, and now and then two
    #: source frames apart where the recorder missed one; the slot count
    #: predicts neither.  With this on, the scene's own clock is recovered from
    #: the moving balls instead (see ``billiards.clock``).  It only engages on a
    #: file that has repeated a frame while a ball was moving, which a genuine
    #: constant-rate recording never does, so such a recording is untouched.
    source_clock: bool = True
    #: Repeats-while-moving it takes to engage.
    source_clock_min_moving_repeats: int = 3
    #: The most source frames one measured frame may be found to span.
    source_clock_max_skip: int = 3
    #: A ball's motion is only used as evidence if one more source frame moves
    #: it at least this many ball radii further -- slower than that, and
    #: measurement noise cannot tell one frame from two.
    source_clock_min_step_ball_radii: float = 0.5

    #: After a cut, how many consecutive frames must agree on the new table
    #: before it is adopted.  One frame during a crossfade is a poor basis for
    #: a homography that everything downstream depends on.  The periodic
    #: re-check asks the same of a table that seems to have moved: one frame's
    #: outline, with a player over the near rail or the cushion tops half in
    #: it, rebuilt the 2026 Premier League final's table every 5 s.
    recovery_frames: int = 5

    #: How far, in pixels, those frames' corners may disagree and still count as
    #: agreement.  Above this the view is still settling (a pan, a dissolve).
    recovery_max_spread_px: float = 12.0

    #: A candidate table polygon is only adopted if at least this fraction of it
    #: is cloth-coloured.  A crowd shot will always yield *some* quadrilateral
    #: from *some* blob; this is what distinguishes a table from a sponsor
    #: banner, and stops a bad recalibration from overwriting a good one.
    min_bed_coverage: float = 0.72

    #: Shrink the detected table polygon by this many ball diameters before
    #: looking for balls, so cushions/rails/pocket jaws do not create blobs.
    bed_margin_ball_diameters: float = 0.35

    #: Move the fitted edges in from the outline of the cloth to the cushion
    #: noses (see ``table.refine_to_cushion_noses``).  The cushion tops are
    #: clothed, so on the sample broadcasts the outline ran two inches outside
    #: the long rails' noses and over the far cushion's face.
    fit_cushion_noses: bool = True
    #: How far inside the outline to look for a nose, in inches.  Deep enough
    #: for a whole far cushion, top and face: when the line under its nose is
    #: faint -- a pixel wide at 480p -- the cloth outline runs over all of it,
    #: 6.5 in past the nose on the synthetic 480p clip, which a 6 in search
    #: never reached.
    cushion_nose_search_in: float = 10.0
    #: The smallest colour step (weighted Lab, as for ball colours) that
    #: counts as the edge of the bed.  The nose line on the sample clips is a
    #: step of 20-40 from bed that varies by 1-2.
    cushion_nose_min_step: float = 10.0
    #: Nose height, in ball diameters: 63.5% on a regulation pool table.
    cushion_nose_height_ball_diameters: float = 0.635

    #: Place each ball from its centre, a radius above the cloth, rather than
    #: as if it were painted on the cloth.  Needs the camera, which is
    #: recovered from the table's homography; on the sample broadcasts the
    #: correction is 2.4 in at the near rail and 4.0 in at the far one.
    ball_parallax: bool = True

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

    #: Ceilings.  Without these, a frame where the background shares the cloth's
    #: hue -- blue banners behind a blue table, which is most tournament
    #: footage -- gives a robust spread so wide the "cloth" window accepts the
    #: entire image.  A real cloth is one colour; these bound how far that can
    #: be stretched before the estimate is simply wrong.
    max_hue_halfwidth: float = 22.0
    max_sat_halfwidth: float = 75.0
    max_val_halfwidth: float = 85.0

    #: Rounds of re-estimation restricted to the region the previous estimate
    #: selected.  The first pass sees the whole frame, background included; each
    #: refinement re-measures using only pixels inside the largest region that
    #: survived, which converges onto the bed itself.
    refine_iterations: int = 2

    #: The value window is deliberately asymmetric.  A shadow can only make the
    #: cloth *darker*, never brighter, and the shadow a ball casts is what welds
    #: neighbouring balls into one blob if it is treated as foreground.  Hue and
    #: saturation stay tight, so widening downwards costs almost nothing: a dark
    #: ball is excluded by its hue or its low saturation, not by its brightness.
    shadow_value_factor: float = 2.2

    #: Hard floor on that downward extension, as a fraction of the measured
    #: cloth value.  A cast shadow on cloth is a *multiplicative* darkening and
    #: bottoms out around half the lit value; anything darker is an object, not
    #: a shadow.  Without this floor the widened window swallows dark balls
    #: whole -- on grey cloth, the black ball simply never appears.
    shadow_min_value_ratio: float = 0.55

    #: Pixels darker / less saturated than this are never cloth.  Kills shadow
    #: under the rail and the black bars around letterboxed video.
    min_value: int = 25
    min_saturation: int = 25

    #: Downscale factor used when histogramming frames, for speed.
    analysis_scale: float = 0.5

    #: Besides the commonest hue, this many of the commonest colours (by hue
    #: and saturation) and greys (by brightness) are tried as the cloth, each
    #: covering at least ``candidate_min_share`` of the sampled pixels, and the
    #: most table-like is kept (``table.choose_cloth``).  Needed wherever the
    #: floor or the walls are more of the picture than the bed: the 2026 US
    #: Open's royal-blue floor was taken for its blue-grey cloth.
    colour_candidates: int = 4
    grey_candidates: int = 2
    candidate_min_share: float = 0.015
    #: The commonest hue stays the cloth unless another candidate scores this
    #: many times higher, so footage it already handles is left as it was.
    candidate_margin: float = 1.25


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
    #: A blob below this fraction of the expected ball area is a cushion
    #: sliver, a pocket edge or a piece of chalk, not a ball.  Even a ball
    #: half-hidden behind another still covers about half its area.
    min_area_ratio: float = 0.33
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
    #: Also search where a ball on the bed can *appear*: seen at the height of
    #: its centre, a ball against the far cushion is past the bed's far edge.
    search_raised_bed: bool = False
    #: ...where a detection with more than this fraction of its disc past the
    #: bed's edge can follow a ball already being tracked, but never start a
    #: track of its own.
    raised_band_outside_fraction: float = 0.55

    #: When splitting a cluster, a peak of the distance transform is only a ball
    #: if its height is close to the ball radius.  A wider object (an arm, a
    #: sleeve, a pile of chalk) has a much taller ridge, so this band is what
    #: keeps the generous max_area_ratio above from admitting junk.
    split_peak_min_ratio: float = 0.55
    split_peak_max_ratio: float = 1.75

    #: A split blob has to look like a group of balls in one of two ways, and
    #: a bridge hand on the bed manages neither.  A hand is ten ball-areas of
    #: "not cloth" whose knuckles are ball-thick and roughly ball-sized, so the
    #: splitter happily reports three balls inside it -- and on broadcast
    #: footage one of them wins the CUE label from the real cue ball.
    #:
    #: The first way is arithmetic: a group of touching balls *is* a union of
    #: discs of the known radius, so once the discs are placed, the blob's
    #: ball-thick core (distance transform above half a ball radius) should be
    #: covered.  This is the fraction of that core the discs must explain.
    #:
    #: The second way exists because failing the first does not prove the blob
    #: is not balls -- it may be balls the splitter could not separate.  A
    #: racked triangle of same-coloured neighbours yields five of eight, so its
    #: discs cover 0.60 of it; rejecting on that alone cost 9 points of recall
    #: on the ground-truth clip.  See ``cluster_rim_contrast_min``.
    cluster_core_coverage_min: float = 0.85

    #: ...the second way: the discs sit on unmistakable ball edges.  This is
    #: the smallest *median* ``rim_contrast`` across a blob's candidates for it
    #: to be believed despite not accounting for itself.  A blob is a clump of
    #: balls or it is not -- one ball and two knuckles is not a thing -- so the
    #: whole blob is judged together.
    #:
    #: Measured over four clips: an under-split rack scores 64-115 and a
    #: well-split pair 62-65, against 16-46 for a hand.  The threshold sits in
    #: that gap.
    cluster_rim_contrast_min: float = 55.0

    #: ...but a blob of two or more candidates out in the open, clear of the
    #: bed's edge, only needs this.  A hand, forearm or cue reaches the table
    #: from outside and so crosses the edge; a static rack does not.  Measured:
    #: rejected blobs in the open on the sample clips 6-13, a camera-realistic
    #: rack 33-51.
    cluster_open_rim_contrast_min: float = 30.0

    #: A ball *ends* at its rim: one ball-radius out, the colour has to change,
    #: because what surrounds a ball is cloth or another ball and never more of
    #: itself.  This is the smallest median colour step (weighted CIE Lab, the
    #: same metric the tracker matches identities with) across the rim, taken
    #: over directions so a partly occluded ball still passes.
    #:
    #: A disc drawn anywhere inside a hand, a forearm or a shirt fails it: the
    #: material simply continues.  Set to 0 to disable.
    rim_contrast_min: float = 12.0

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

    #: Coasting is extrapolation, so how far a track may be extrapolated is
    #: capped by how much evidence built its motion model: this many frames of
    #: coasting per frame the track was actually observed, up to
    #: ``max_age_coasting``.
    #:
    #: Without it, a blob that happened to look like a ball three frames
    #: running is confirmed and then draws a confident trajectory for the full
    #: coasting window on the strength of nothing -- on real clips the survivors
    #: were tracks with six observations and sixty frames of invented path.  A
    #: ball that has been watched for hundreds of frames is unaffected, which is
    #: the point: it has earned the benefit of the doubt.
    coast_frames_per_hit: float = 1.0

    #: A ball is treated as stationary below this speed, which suppresses
    #: trajectory jitter when nothing is moving.
    stationary_speed_in_s: float = 4.0

    #: A track that dies waits this long in case its ball reappears -- out of
    #: the pocket jaws, or from behind the player -- within this many ball
    #: diameters of where it vanished, looking the same.  It then gets its
    #: identity back and its death (and any pot) is withdrawn.  Pots are
    #: therefore reported this much later, at the frame the ball vanished.
    revive_window_s: float = 1.5
    revive_distance_ball_diameters: float = 3.0

    #: A broadcast cuts away from the table -- to a player, a replay, another
    #: camera -- and comes back to the same balls.  Until 28 Sep 2026 every
    #: ball then became a new one: a minute of the 2025 Mosconi Cup had 91
    #: "balls" for 10.  So at a cut every ball is set aside, and when the
    #: table is back each is re-found: a ball that stood still is where it
    #: was, within this many ball diameters (two camera angles place a ball
    #: an inch or two apart)...
    reclaim_distance_ball_diameters: float = 2.0
    #: ...and one that rolled while the camera was away is re-found anywhere
    #: by its colour, if it is within this fraction of
    #: ``max_color_distance`` and no other ball set aside is nearly as close.
    reclaim_colour_ratio: float = 0.55
    #: A ball set aside waits this long, counting only time with the table in
    #: view, before it is given up for lost (or potted, if it was heading for
    #: a pocket when the camera cut away).
    reclaim_window_s: float = 4.0


@dataclass
class EventConfig:
    """Collision / cushion / pot detection, all in physical units."""

    #: Ball-ball contact when the two came within this many ball diameters of
    #: each other during the frame (see ``EventDetector._collisions``, which
    #: measures the closest approach along their paths rather than sampling the
    #: gap at the end of the frame).
    #:
    #: Balls that touch are exactly 1.0 diameters apart, so everything above
    #: that is the measurement-error budget, and it has to be: the estimate
    #: comes from two filtered paths, and smoothing rounds off the corner at
    #: contact, so a real contact never quite reads as one.  Measured -- real
    #: contacts read 0.80-1.12 on the three sample clips and 0.97-1.10 on the
    #: synthetic clip; near misses that ground truth puts 1.46-1.50 diameters
    #: apart read 1.40-1.51.  This sits in the gap between the two.
    #:
    #: It used to sit at 1.12, on the floor of that gap rather than in it,
    #: which is why the cue ball on ``fedor_shot.mp4`` potted a ball and
    #: reported no collision: the contact measured 1.1205.
    contact_distance_ball_diameters: float = 1.25

    #: Minimum speed at which the gap between two balls has to be closing for
    #: their contact to count as a collision rather than as two balls resting
    #: against each other -- those are as close as it gets, forever.
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

    #: A cushion contact puts the ball's centre one radius off the cushion's
    #: nose; a turn-round within this many *further* radii of the calibrated
    #: rail line counts.  It is wide because the calibrated outline is the
    #: outline of the cloth, and on a real table the cushions are clothed too:
    #: from behind an end rail the camera sees the far cushion's face and the
    #: long cushions' tops, so on those rails the nose is up to ~4 in inside
    #: the fitted edge.  Measured on fedor_shot.mp4: the near-rail bounce
    #: turns round 0.95 in off it, the long-rail one 3.4 in, the far-rail one
    #: ~4.9 in.  A turn-round next to another ball is never a cushion.
    cushion_contact_tolerance_ball_radii: float = 4.5
    #: ...but not within this many ball diameters of a pocket, where a ball
    #: rattling in the jaws or dropping in turns a corner too.
    cushion_pocket_clearance_ball_diameters: float = 1.2
    #: A ball seen all but still for this long since it last closed on a rail
    #: is no longer approaching it.  Steps too small to class leave a ball's
    #: "approaching" state alone, so without this a ball that rolled toward a
    #: rail, stopped short and was knocked away seconds later "bounced" off a
    #: rail 39 in away (fedor_shot, frame 303).  Real bounces on the sample
    #: and synthetic clips were closing 0.02-0.4 s before -- or were unseen in
    #: between, which does not count: albin_fedor's cue ball spends 0.6 s
    #: hidden in a pocket's jaws before it comes off the rail.
    cushion_approach_max_age_s: float = 0.6
    #: Two balls' corners are one collision if they are this close in time...
    kink_pair_window_s: float = 0.08
    #: ...and their corners this many ball diameters apart (touching is 1.0).
    kink_pair_distance_ball_diameters: float = 1.5
    #: One ball cannot turn two corners closer together than this.
    kink_refractory_s: float = 0.1


@dataclass
class BallsConfig:
    """Naming balls by the number printed on them.  See ``billiards.balls``."""

    #: Label balls with their numbers.  Off, a ball is its tracker id (``#5``)
    #: unless it is the cue ball or the 8.
    enabled: bool = True
    #: ``standard`` (4 purple, 5 orange, white-capped stripes), ``tv`` (the
    #: set in the sample broadcasts: 4 pink, 5 purple, black-capped stripes),
    #: or ``auto``, which keeps whichever of the two fits the table better.
    ball_set: str = "auto"
    #: The numbers that can be on the table: ``1-9`` for 9-ball, ``1-10`` for
    #: 10-ball, ``1-15`` for 8-ball or when unsure.
    numbers: str = "1-15"
    #: A ball whose best available number costs more than this -- about this
    #: many typical deviations off in hue, lightness, saturation or stripe --
    #: stays unnumbered rather than being named wrongly.
    max_cost: float = 3.0
    #: How much better a different number has to fit before a ball changes
    #: number, in the same units.
    stickiness: float = 0.6
    #: Detections of a ball on its own -- not split out of a cluster, which
    #: samples the neighbours too -- before it is numbered.  Its colour is an
    #: average that takes a few of those to settle.
    min_colour_samples: int = 4


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

    #: Where the top-down diagram goes.
    #:
    #: ``below`` puts it in a bar under the video, with the status text beside
    #: it, and nothing is drawn over the picture at all.  The camera view of a
    #: pool table fills its frame -- on the sample clips the bed covers the
    #: whole lower half -- so an inset in any corner sits on top of the table
    #: it is describing, which is exactly where the viewer is looking.
    #:
    #: ``inset`` is the old in-frame corner, for when the output has to keep
    #: the source resolution.
    overhead_panel_place: str = "below"
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
    balls: BallsConfig = field(default_factory=BallsConfig)
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
        "balls": BallsConfig,
        "render": RenderConfig,
    }
)
