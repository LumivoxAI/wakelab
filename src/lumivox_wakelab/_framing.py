"""Internal native-frame coordinators over retained PCM ranges."""

from __future__ import annotations

import math
from typing import Protocol
from numbers import Real
from dataclasses import dataclass

from ._timeline import SampleRange, AudioTimeline
from .vad._backend import VadBackend
from .wakeword._backend import WakeWordBackend

_SAMPLE_RATE = 16_000


class _FrameBackend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: object, /) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One finite backend result associated with its exact input range."""

    sample_range: SampleRange
    value: float


class _Framer:
    def __init__(self, backend: _FrameBackend, /) -> None:
        sample_rate = backend.sample_rate
        frame_samples = backend.frame_samples
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate != _SAMPLE_RATE:
            raise ValueError(f"backend sample_rate must be {_SAMPLE_RATE}")
        if isinstance(frame_samples, bool) or not isinstance(frame_samples, int) or frame_samples <= 0:
            raise ValueError("backend frame_samples must be a positive integer")
        self._backend = backend
        self._frame_samples = frame_samples

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def _infer(self, timeline: AudioTimeline, start: int) -> FrameResult:
        end = start + self._frame_samples
        value: object = self._backend.infer(timeline.read(start, end))
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("backend infer must return a finite scalar")
        return FrameResult(SampleRange(start, end), float(value))

    def _validate_available_range(self, timeline: AudioTimeline, start: int, end: int) -> None:
        retained = timeline.retained_range
        if start < retained.start or end > retained.end:
            raise ValueError("frame range must be retained")
        if end < start:
            raise ValueError("frame range must be ordered")


class VadFramer(_Framer):
    """Consume every complete VAD frame in one continuity segment."""

    def __init__(self, backend: VadBackend, /) -> None:
        super().__init__(backend)
        self._position: int | None = None

    @property
    def position(self) -> int | None:
        return self._position

    def start_segment(self, start: int, /) -> None:
        """Reset the backend and align the next frame to a new segment."""

        if start < 0:
            raise ValueError("segment start must be nonnegative")
        self._backend.reset()
        self._position = start

    def consume(self, timeline: AudioTimeline, end: int, /) -> list[FrameResult]:
        """Infer all complete frames before ``end`` and leave its tail unscored."""

        if self._position is None:
            raise RuntimeError("VAD segment has not been started")
        self._validate_available_range(timeline, self._position, end)
        results: list[FrameResult] = []
        while self._position + self._frame_samples <= end:
            result = self._infer(timeline, self._position)
            self._position = result.sample_range.end
            results.append(result)
        return results


class WakeWordFramer(_Framer):
    """Mechanically frame one contiguous wake-word candidate at a time."""

    def __init__(self, backend: WakeWordBackend, /) -> None:
        super().__init__(backend)
        self._position: int | None = None

    @property
    def position(self) -> int | None:
        return self._position

    @property
    def evaluating(self) -> bool:
        return self._position is not None

    def start_candidate(self, start: int, /) -> None:
        """Reset model history and begin a contiguous candidate at ``start``."""

        if self._position is not None:
            raise RuntimeError("wake-word candidate is already active")
        if start < 0:
            raise ValueError("candidate start must be nonnegative")
        self._backend.reset()
        self._position = start

    def consume(self, timeline: AudioTimeline, end: int, /) -> list[FrameResult]:
        """Infer complete contiguous candidate frames before ``end``."""

        if self._position is None:
            raise RuntimeError("wake-word candidate has not been started")
        self._validate_available_range(timeline, self._position, end)
        results: list[FrameResult] = []
        while self._position + self._frame_samples <= end:
            result = self._infer(timeline, self._position)
            self._position = result.sample_range.end
            results.append(result)
        return results

    def end_candidate(self, end: int, /) -> None:
        """Discard an unscored candidate tail after validating its contiguous end."""

        if self._position is None:
            raise RuntimeError("wake-word candidate has not been started")
        if end < self._position:
            raise ValueError("candidate end precedes its consumed frames")
        self._position = None

    def reset(self) -> None:
        """Cancel any candidate and restore backend history at a hard boundary."""

        self._backend.reset()
        self._position = None
