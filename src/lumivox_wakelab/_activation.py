"""Wake activation lifecycle independent from model and output ownership."""

from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass

from ._timeline import SampleRange
from .vad._policy import VadRange
from .wakeword._policy import WakeDecision


class ActivationState(StrEnum):
    """The wake activation state for one continuity segment."""

    ARMED = "armed"
    EVALUATING = "evaluating"
    ACTIVATED = "activated"


@dataclass(frozen=True, slots=True)
class ActivationRange:
    """One nonempty VAD range with independent activation metadata."""

    sample_range: SampleRange
    is_speech: bool
    is_activated: bool


class ActivationLifecycle:
    """Apply confirmed wake decisions without changing speech metadata."""

    def __init__(self) -> None:
        self._position: int | None = None
        self._state = ActivationState.ARMED
        self._trigger: WakeDecision | None = None
        self._rearm_boundary: int | None = None

    @property
    def state(self) -> ActivationState:
        """Return the current conceptual activation state."""

        return self._state

    @property
    def wake_inference_bypassed(self) -> bool:
        """Whether the wake backend must be bypassed for subsequent audio."""

        return self._state is ActivationState.ACTIVATED

    @property
    def trigger(self) -> WakeDecision | None:
        """Return the confirmed trigger retained for diagnostics, if any."""

        return self._trigger

    def start_segment(self, start: int, /) -> None:
        """Start a continuity segment while preserving an activation latch."""

        self._validate_position("segment start", start)
        if self._position is not None:
            raise RuntimeError("activation segment is already active")
        self._position = start
        if self._state is ActivationState.EVALUATING:
            self._state = ActivationState.ARMED

    def consume(
        self,
        ranges: list[VadRange],
        decision: WakeDecision | None,
        /,
        *,
        evaluating: bool,
    ) -> list[ActivationRange]:
        """Apply a decision to contiguous finalized VAD ranges.

        A decision must start within this batch, which lets the caller retain the
        affected audio until its score policy has enough evidence to confirm it.
        """

        if self._position is None:
            raise RuntimeError("activation segment has not been started")
        if not isinstance(evaluating, bool):
            raise TypeError("evaluating must be a bool")
        end = self._validate_ranges(ranges)
        if decision is not None and not isinstance(decision, WakeDecision):
            raise TypeError("decision must be a WakeDecision or None")
        if decision is not None:
            if self._state is ActivationState.ACTIVATED and (
                self._rearm_boundary is None or decision.activation_start < self._rearm_boundary
            ):
                raise RuntimeError("activation is already latched")
            if not self._position <= decision.activation_start <= end:
                raise ValueError("activation start must be within consumed ranges")

        output: list[ActivationRange] = []
        activation_start = decision.activation_start if decision is not None else None
        for vad_range in ranges:
            sample_range = vad_range.sample_range
            boundaries = [sample_range.start, sample_range.end]
            if activation_start is not None and sample_range.start < activation_start < sample_range.end:
                boundaries.append(activation_start)
            if self._rearm_boundary is not None and sample_range.start < self._rearm_boundary < sample_range.end:
                boundaries.append(self._rearm_boundary)
            boundaries.sort()
            for start, finish in zip(boundaries, boundaries[1:]):
                before_rearm = self._rearm_boundary is None or start < self._rearm_boundary
                activated = (before_rearm and self._state is ActivationState.ACTIVATED) or (
                    activation_start is not None and start >= activation_start
                )
                self._append(output, start, finish, vad_range.is_speech, activated)

        self._position = end
        if decision is not None:
            self._state = ActivationState.ACTIVATED
            self._trigger = decision
            self._rearm_boundary = None
        elif self._rearm_boundary is not None and end >= self._rearm_boundary:
            self._state = ActivationState.ARMED
            self._rearm_boundary = None
        elif self._state is not ActivationState.ACTIVATED:
            self._state = ActivationState.EVALUATING if evaluating else ActivationState.ARMED
        return output

    def rearm(self, boundary: int, /) -> None:
        """Clear activation at the next not-yet-accepted sample boundary."""

        self._validate_position("re-arm boundary", boundary)
        if self._position is None:
            self._state = ActivationState.ARMED
            self._trigger = None
            self._rearm_boundary = None
            return
        if boundary < self._position:
            raise ValueError("re-arm boundary cannot precede the next unconsumed sample")
        if boundary == self._position:
            self._state = ActivationState.ARMED
            self._rearm_boundary = None
        else:
            self._rearm_boundary = boundary
        self._trigger = None

    def finish_segment(self, end: int, /) -> None:
        """Reject unresolved evaluation at an EOS, boundary, or terminal failure."""

        if self._position is None:
            raise RuntimeError("activation segment has not been started")
        self._validate_position("segment end", end)
        if end != self._position:
            raise ValueError("segment end must match consumed ranges")
        if self._state is ActivationState.EVALUATING:
            self._state = ActivationState.ARMED
        self._position = None
        self._rearm_boundary = None

    def _validate_ranges(self, ranges: list[VadRange]) -> int:
        assert self._position is not None
        position = self._position
        for vad_range in ranges:
            if not isinstance(vad_range, VadRange):
                raise TypeError("ranges must contain VadRange values")
            sample_range = vad_range.sample_range
            if sample_range.start != position or sample_range.end <= position:
                raise ValueError("VAD ranges must be contiguous nonempty ranges")
            if not isinstance(vad_range.is_speech, bool):
                raise TypeError("VadRange.is_speech must be a bool")
            position = sample_range.end
        return position

    @staticmethod
    def _validate_position(name: str, position: object) -> None:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(f"{name} must be an integer")
        if position < 0:
            raise ValueError(f"{name} must be nonnegative")

    @staticmethod
    def _append(output: list[ActivationRange], start: int, end: int, is_speech: bool, is_activated: bool) -> None:
        if start >= end:
            return
        if (
            output
            and output[-1].sample_range.end == start
            and output[-1].is_speech == is_speech
            and output[-1].is_activated == is_activated
        ):
            previous = output[-1]
            output[-1] = ActivationRange(SampleRange(previous.sample_range.start, end), is_speech, is_activated)
            return
        output.append(ActivationRange(SampleRange(start, end), is_speech, is_activated))
