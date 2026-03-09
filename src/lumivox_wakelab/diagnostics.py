"""Immutable opt-in diagnostics for streaming wake-word processing."""

from __future__ import annotations

import math
from enum import StrEnum
from dataclasses import dataclass


def _position(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _range(start: int, end: int) -> None:
    _position("start_sample", start)
    _position("end_sample", end)
    if end <= start:
        raise ValueError("diagnostic sample range must be nonempty")


def _value(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


class VadTransitionReason(StrEnum):
    """Why a finalized VAD state transition became observable."""

    EVIDENCE = "evidence"
    SEGMENT_END = "segment_end"
    PROCESSING_FAILURE = "processing_failure"


class WakeCandidateEndReason(StrEnum):
    """Why wake-word evaluation of a candidate ended."""

    SILENCE_BRIDGE_EXCEEDED = "silence_bridge_exceeded"
    ACTIVATED = "activated"
    REARMED = "rearmed"
    CONTINUITY_BOUNDARY = "continuity_boundary"
    FINISHED = "finished"
    PROCESSING_FAILED = "processing_failed"


class ContinuityBoundaryReason(StrEnum):
    """Source or application condition that split continuity."""

    INPUT_DISCONTINUITY = "input_discontinuity"
    GENERATION_CHANGE = "generation_change"
    GENERATION_CHANGE_AND_DISCONTINUITY = "generation_change_and_discontinuity"
    EXPLICIT_DISCONTINUE = "explicit_discontinue"


class DiagnosticFailureOperation(StrEnum):
    """Processor operation that failed."""

    PROCESS = "process"
    FINALIZE = "finalize"
    VAD_CLOSE = "vad_close"
    WAKE_CLOSE = "wake_close"


@dataclass(frozen=True, slots=True)
class VadFrameDiagnostic:
    generation: int
    start_sample: int
    end_sample: int
    probability: float

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _range(self.start_sample, self.end_sample)
        _value("probability", self.probability)
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VadSpeechDiagnostic:
    generation: int
    sample: int
    is_speech: bool
    reason: VadTransitionReason

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _position("sample", self.sample)
        if not isinstance(self.is_speech, bool):
            raise TypeError("is_speech must be a bool")
        if not isinstance(self.reason, VadTransitionReason):
            raise TypeError("reason must be a VadTransitionReason")


@dataclass(frozen=True, slots=True)
class WakeCandidateStartedDiagnostic:
    generation: int
    start_sample: int

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _position("start_sample", self.start_sample)


@dataclass(frozen=True, slots=True)
class WakeCandidateEndedDiagnostic:
    generation: int
    start_sample: int
    end_sample: int
    reason: WakeCandidateEndReason

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _position("start_sample", self.start_sample)
        _position("end_sample", self.end_sample)
        if self.end_sample < self.start_sample:
            raise ValueError("candidate end cannot precede its start")
        if not isinstance(self.reason, WakeCandidateEndReason):
            raise TypeError("reason must be a WakeCandidateEndReason")


@dataclass(frozen=True, slots=True)
class WakeFrameDiagnostic:
    generation: int
    start_sample: int
    end_sample: int
    score: float
    threshold_met: bool
    consecutive_hit_count: int

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _range(self.start_sample, self.end_sample)
        _value("score", self.score)
        if not isinstance(self.threshold_met, bool):
            raise TypeError("threshold_met must be a bool")
        _position("consecutive_hit_count", self.consecutive_hit_count)


@dataclass(frozen=True, slots=True)
class ActivationDiagnostic:
    generation: int
    activation_start_sample: int
    first_start_sample: int
    first_end_sample: int
    first_score: float
    confirming_start_sample: int
    confirming_end_sample: int
    confirming_score: float

    def __post_init__(self) -> None:
        _position("generation", self.generation)
        _position("activation_start_sample", self.activation_start_sample)
        _range(self.first_start_sample, self.first_end_sample)
        _value("first_score", self.first_score)
        _range(self.confirming_start_sample, self.confirming_end_sample)
        _value("confirming_score", self.confirming_score)


@dataclass(frozen=True, slots=True)
class RearmDiagnostic:
    generation: int | None
    effective_sample: int

    def __post_init__(self) -> None:
        if self.generation is not None:
            _position("generation", self.generation)
        _position("effective_sample", self.effective_sample)


@dataclass(frozen=True, slots=True)
class ContinuityBoundaryDiagnostic:
    sample: int
    previous_generation: int | None
    next_generation: int | None
    reason: ContinuityBoundaryReason

    def __post_init__(self) -> None:
        _position("sample", self.sample)
        if self.previous_generation is not None:
            _position("previous_generation", self.previous_generation)
        if self.next_generation is not None:
            _position("next_generation", self.next_generation)
        if not isinstance(self.reason, ContinuityBoundaryReason):
            raise TypeError("reason must be a ContinuityBoundaryReason")


@dataclass(frozen=True, slots=True)
class FailureDiagnostic:
    generation: int | None
    retained_end_sample: int
    accepted_end_sample: int
    operation: DiagnosticFailureOperation
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if self.generation is not None:
            _position("generation", self.generation)
        _position("retained_end_sample", self.retained_end_sample)
        _position("accepted_end_sample", self.accepted_end_sample)
        if self.accepted_end_sample < self.retained_end_sample:
            raise ValueError("accepted_end_sample cannot precede retained_end_sample")
        if not isinstance(self.operation, DiagnosticFailureOperation):
            raise TypeError("operation must be a DiagnosticFailureOperation")
        if not isinstance(self.error_type, str) or not isinstance(self.message, str):
            raise TypeError("failure text must be strings")


type DiagnosticEvent = (
    VadFrameDiagnostic
    | VadSpeechDiagnostic
    | WakeCandidateStartedDiagnostic
    | WakeCandidateEndedDiagnostic
    | WakeFrameDiagnostic
    | ActivationDiagnostic
    | RearmDiagnostic
    | ContinuityBoundaryDiagnostic
    | FailureDiagnostic
)


@dataclass(frozen=True, slots=True)
class DiagnosticDrain:
    """Events and overflow count accumulated since the previous drain."""

    events: tuple[DiagnosticEvent, ...]
    dropped_events: int

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple")
        _position("dropped_events", self.dropped_events)


__all__ = [
    "ActivationDiagnostic",
    "ContinuityBoundaryDiagnostic",
    "ContinuityBoundaryReason",
    "DiagnosticDrain",
    "DiagnosticEvent",
    "DiagnosticFailureOperation",
    "FailureDiagnostic",
    "RearmDiagnostic",
    "VadFrameDiagnostic",
    "VadSpeechDiagnostic",
    "VadTransitionReason",
    "WakeCandidateEndedDiagnostic",
    "WakeCandidateEndReason",
    "WakeCandidateStartedDiagnostic",
    "WakeFrameDiagnostic",
]
