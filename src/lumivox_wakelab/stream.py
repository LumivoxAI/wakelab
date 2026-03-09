"""Public values and lifecycle contract for streaming wake-word processing."""

from __future__ import annotations

import math
from typing import Protocol
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .diagnostics import DiagnosticDrain

_PCM16_DTYPE = np.dtype("<i2")


def _validate_samples(samples: object) -> None:
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a numpy.ndarray")
    if samples.dtype != _PCM16_DTYPE:
        raise TypeError('samples must have dtype numpy.dtype("<i2")')
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("samples must not be empty")
    if not samples.flags.writeable:
        raise ValueError("samples must be writable")


def _validate_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_positive_integer(name: str, value: object) -> None:
    _validate_nonnegative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _validate_boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_probability(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class InputChunk:
    """One caller-owned chunk borrowed for the duration of ``process``."""

    samples: NDArray[np.int16]
    running_time_ns: int
    captured_at_ns: int
    generation: int
    discontinuity: bool

    def __post_init__(self) -> None:
        _validate_samples(self.samples)
        _validate_nonnegative_integer("running_time_ns", self.running_time_ns)
        _validate_nonnegative_integer("captured_at_ns", self.captured_at_ns)
        _validate_nonnegative_integer("generation", self.generation)
        _validate_boolean("discontinuity", self.discontinuity)


@dataclass(frozen=True, slots=True)
class OutputChunk:
    """Receiver-owned audio with uniform speech and activation metadata."""

    samples: NDArray[np.int16]
    running_time_ns: int
    captured_at_ns: int
    generation: int
    discontinuity: bool
    is_speech: bool
    is_activated: bool

    def __post_init__(self) -> None:
        _validate_samples(self.samples)
        _validate_nonnegative_integer("running_time_ns", self.running_time_ns)
        _validate_nonnegative_integer("captured_at_ns", self.captured_at_ns)
        _validate_nonnegative_integer("generation", self.generation)
        _validate_boolean("discontinuity", self.discontinuity)
        _validate_boolean("is_speech", self.is_speech)
        _validate_boolean("is_activated", self.is_activated)


@dataclass(frozen=True, slots=True)
class VadPolicyConfig:
    """Explicit sample-level temporal policy for voice activity."""

    speech_threshold: float
    silence_threshold: float
    minimum_speech_samples: int
    minimum_silence_samples: int
    left_padding_samples: int
    right_padding_samples: int

    def __post_init__(self) -> None:
        _validate_probability("speech_threshold", self.speech_threshold)
        _validate_probability("silence_threshold", self.silence_threshold)
        if self.silence_threshold >= self.speech_threshold:
            raise ValueError("silence_threshold must be less than speech_threshold")
        _validate_positive_integer("minimum_speech_samples", self.minimum_speech_samples)
        _validate_positive_integer("minimum_silence_samples", self.minimum_silence_samples)
        _validate_nonnegative_integer("left_padding_samples", self.left_padding_samples)
        _validate_nonnegative_integer("right_padding_samples", self.right_padding_samples)


@dataclass(frozen=True, slots=True)
class WakePolicyConfig:
    """Explicit score and candidate policy for one wake-word classifier."""

    score_threshold: float
    consecutive_score_count: int
    silence_bridge_samples: int
    pre_roll_samples: int

    def __post_init__(self) -> None:
        _validate_probability("score_threshold", self.score_threshold)
        _validate_positive_integer("consecutive_score_count", self.consecutive_score_count)
        _validate_nonnegative_integer("silence_bridge_samples", self.silence_bridge_samples)
        _validate_nonnegative_integer("pre_roll_samples", self.pre_roll_samples)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Complete public policy and steady-state retention configuration."""

    vad_policy: VadPolicyConfig
    wake_policy: WakePolicyConfig
    max_retained_audio_samples: int

    def __post_init__(self) -> None:
        if not isinstance(self.vad_policy, VadPolicyConfig):
            raise TypeError("vad_policy must be a VadPolicyConfig")
        if not isinstance(self.wake_policy, WakePolicyConfig):
            raise TypeError("wake_policy must be a WakePolicyConfig")
        _validate_positive_integer("max_retained_audio_samples", self.max_retained_audio_samples)


class StreamProcessor(Protocol):
    """Synchronous lifecycle contract implemented by the future orchestrator."""

    def process(self, chunk: InputChunk, /) -> list[OutputChunk]: ...

    def rearm(self) -> list[OutputChunk]: ...

    def discontinue(self) -> list[OutputChunk]: ...

    def finish(self) -> list[OutputChunk]: ...

    def drain_failed(self) -> list[OutputChunk]: ...

    def drain_diagnostics(self) -> DiagnosticDrain: ...

    def close(self) -> list[OutputChunk]: ...


class StreamError(RuntimeError):
    """Base class for streaming processor failures."""


class StreamStateError(StreamError):
    """A lifecycle operation was attempted in an invalid processor state."""


class StreamProcessingError(StreamError):
    """Processing failed after the complete input chunk was accepted."""


class StreamCloseError(StreamError):
    """Resource closure failed after the processor finalized retained audio."""

    def __init__(
        self,
        message: str,
        outputs: list[OutputChunk],
        errors: tuple[BaseException, ...],
    ) -> None:
        super().__init__(message)
        self.outputs = outputs
        self.errors = errors


__all__ = [
    "InputChunk",
    "OutputChunk",
    "StreamCloseError",
    "StreamConfig",
    "StreamError",
    "StreamProcessingError",
    "StreamProcessor",
    "StreamStateError",
    "VadPolicyConfig",
    "WakePolicyConfig",
]
