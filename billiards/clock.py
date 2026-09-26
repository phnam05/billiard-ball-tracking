"""How much *scene* time passed between two measured frames.

The obvious answer -- frames elapsed divided by the file's frame rate -- is
wrong for most footage that reaches this tool.  A broadcast watched in a
browser and screen-recorded at 37.5 fps holds 25 fps content: every source
frame is repeated in one, two or three slots of the file, and now and then one
is never shown at all.  Nothing in the file records which.

Measured on the three sample clips, the slot count between two new frames
says nothing about the time between them.  A ball rolling at constant speed
moves *the same distance* after a one-slot gap as after a two-slot gap (median
ratio 0.96 where the slot clock predicts 2.0), because each new frame is simply
the next source frame, 40 ms later.  Believing the slot clock told the filter
27 ms and then 53 ms in alternation, which is where the "unfixable" residual
speed jitter came from.  About one step in seven covers two source frames --
the recorder missed one -- and shows exactly twice the travel.

So on such a clip the clock is recovered from the scene itself.  Every moving
ball is a clock: its motion model says where it will be after one source
frame, or two, or three, and the detections say which of those happened.
Where one count fits decisively better than the rest, that is what happened,
and the source frame rate is the total of those counts over the slots they
spanned, snapped to the nearest broadcast standard.  Where nothing is moving
fast enough to tell, the frame is timed as the number of source frames its
slots should hold at that rate.  Only decisive frames move the estimate: a tie
counted as evidence is a feedback loop (see ``_DECISIVE_FACTOR``).

Positions alone cannot say how long a source frame is -- a ball twice as fast
filmed half as often looks the same -- only how many of them each step spans.
The count is judged against the balls' velocities, which are in whatever time
base the filters were fed, so when the rate estimate changes those velocities
must change with it (``TrackingPipeline`` rescales them); otherwise, for the
few frames a filter takes to catch up, a one-frame step reads as two, the
estimate climbs and the next steps read longer still.  That loop read 47 and
then 58 fps off 25 fps content on a four-second synthetic clip.  Counting also
bounds the rate: a file that repeats frames while balls move cannot hold more
source frames a second than it has slots, and every new frame is at least one
source frame, so the rate is at least the new-frame rate -- and, with at most
``_MAX_DROP_FRACTION`` of source frames missed, not far above it.

On a short clip the *rate* is the weaker half of this.  ``fedor_shot.mp4``
has a 113-slot stretch of play where nearly every frame is decisive, and it
reads 29.7 fps; the other two sample clips have fewer decisive frames and
land anywhere from 24 to 33.  Which frames skipped a source frame is known
far better than exactly how long a source frame is.

A clip is only treated this way once it has shown frames that repeat *while a
ball is moving*, which a genuinely constant-rate recording never does -- a
moving ball always changes the picture.  Until then, and for every such clip,
the slot clock is used unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

#: Frame rates footage is actually produced at.  An estimate within
#: ``_SNAP_TOLERANCE`` of one of these is taken to be it.  5%, because the
#: frames that decide the rate lean toward the short gaps between new frames
#: -- a step spanning a missed source frame is rarely decisive -- which reads
#: 25 fps content 3-4% fast; the standards are far enough apart that their
#: windows still do not meet (25 * 1.05 < 29.97 * 0.95).
_STANDARD_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
_SNAP_TOLERANCE = 0.05

#: Decisive frames before the source rate estimate replaces the
#: distinct-frame rate it starts from, and before it is snapped to a
#: broadcast standard.
_MIN_PROVISIONAL = 5
_MIN_EVIDENCE = 20

#: A fit is only believed if the best candidate puts the moving balls, on
#: median, within this many ball radii of a detection...
_ACCEPT_BALL_RADII = 0.6
#: ...and the runner-up is at least this many times further off.  Anything
#: closer is a tie, and a tie is not evidence: the frame is timed by the
#: prior, and the rate estimate does not hear about it.  Counting ties is a
#: feedback loop -- as the estimate rises, the prior for a two-slot gap
#: becomes two source frames, ties go to it, and the estimate rises further.
#: That is what read 33 fps off a clip whose decisive frames say 29.
_DECISIVE_FACTOR = 2.5

#: At most this fraction of source frames is taken to be missed by the
#: recorder, which caps the rate at the new-frame rate over one minus it.
#: The sample clips miss about one in seven.
_MAX_DROP_FRACTION = 0.35

#: Fitter: given candidate intervals in seconds, the median prediction error of
#: the moving balls for each (inches), or None when nothing is moving fast
#: enough to tell them apart.
Fitter = Callable[[Sequence[float]], Optional[np.ndarray]]


def snap_rate(fps: float) -> float:
    """``fps``, or the broadcast standard it is within 5% of."""
    best = min(_STANDARD_RATES, key=lambda r: abs(r - fps))
    return best if abs(best - fps) <= _SNAP_TOLERANCE * best else fps


class SourceClock:
    def __init__(
        self,
        container_fps: float,
        ball_radius_in: float,
        enabled: bool = True,
        min_moving_repeats: int = 3,
        max_source_frames: int = 3,
    ) -> None:
        self.container_fps = max(float(container_fps), 1.0)
        self.ball_radius_in = float(ball_radius_in)
        self.enabled = enabled
        self.min_moving_repeats = max(1, int(min_moving_repeats))
        self.max_source_frames = max(1, int(max_source_frames))

        #: Copies seen while a ball was moving: the evidence that this file's
        #: clock is not the scene's.
        self.moving_repeats = 0
        #: Slots and distinct frames seen while a ball was moving, for the
        #: starting estimate of the source rate and the bound on its period.
        self._moving_slots = 0
        self._moving_distinct = 0
        #: Source frames and slots over the frames motion evidence decided.
        self._evidence_k = 0
        self._evidence_slots = 0
        self.evidence_frames = 0
        self.skipped_source_frames = 0

    # -- state -------------------------------------------------------------

    @property
    def retimed(self) -> bool:
        """Has the file shown that its clock is not the scene's?"""
        return self.enabled and self.moving_repeats >= self.min_moving_repeats

    def _distinct_rate(self) -> float:
        """New frames a second while balls moved: the least the source rate
        can be, and what it is when the recorder missed nothing."""
        if self._moving_slots > 0 and self._moving_distinct > 0:
            return self._moving_distinct / self._moving_slots * self.container_fps
        return self.container_fps

    @property
    def source_fps(self) -> float:
        """Best estimate of the rate the scene was filmed at."""
        if not self.retimed:
            return self.container_fps
        lowest = self._distinct_rate()
        if self.evidence_frames >= _MIN_PROVISIONAL and self._evidence_slots > 0:
            rate = self._evidence_k / self._evidence_slots * self.container_fps
            if self.evidence_frames >= _MIN_EVIDENCE:
                rate = snap_rate(rate)
            highest = min(self.container_fps, lowest / (1.0 - _MAX_DROP_FRACTION))
            return float(np.clip(rate, lowest, max(lowest, highest)))
        return lowest

    # -- per frame ---------------------------------------------------------

    def note_repeat(self, moving: bool) -> None:
        """A slot that only repeated the frame before it."""
        if moving:
            self.moving_repeats += 1

    def interval(
        self, slots: int, moving: bool, fit: Optional[Fitter] = None
    ) -> Tuple[float, int]:
        """Scene time since the last measured frame, and source frames it spans.

        ``slots`` is how many slots of the file have passed since then,
        including the repeats that were replayed rather than measured.
        """
        slots = max(1, int(slots))
        if moving:
            self._moving_slots += slots
            self._moving_distinct += 1
        if not self.retimed:
            return slots / self.container_fps, slots

        fps = self.source_fps
        expected = max(1, int(round(slots * fps / self.container_fps)))
        candidates = sorted(set(range(1, self.max_source_frames + 1)) | {expected})

        errors = fit([c / fps for c in candidates]) if fit is not None else None
        if errors is not None and len(errors) >= 2:
            errors = np.asarray(errors, dtype=np.float64)
            order = np.argsort(errors)
            best, runner_up = float(errors[order[0]]), float(errors[order[1]])
            if (
                best <= _ACCEPT_BALL_RADII * self.ball_radius_in
                and runner_up >= _DECISIVE_FACTOR * best
            ):
                k = candidates[int(order[0])]
                self.evidence_frames += 1
                self._evidence_k += k
                self._evidence_slots += slots
                self.skipped_source_frames += k - 1
                return k / fps, k

        # The balls cannot tell -- nothing is moving fast enough, or two
        # candidates fit about equally.  The frame is still timed in whole
        # source frames, as many as the slots it spans should hold at the
        # current rate, and not by the file's clock.  The file's clock is right
        # on average, but mixing it in hands the filter a jittery interval
        # every time the balls are ambiguous; its velocity then drifts, and
        # the next frames' decisions get worse.  Measured, doing that raised
        # the synthetic 95th-percentile speed error from 11 to 17 in/s.
        return expected / fps, expected

    def to_dict(self) -> dict:
        return {
            "retimed": self.retimed,
            "container_fps": round(self.container_fps, 3),
            "source_fps": round(self.source_fps, 3),
            "moving_repeats": self.moving_repeats,
            "evidence_frames": self.evidence_frames,
            "skipped_source_frames": self.skipped_source_frames,
        }
