"""Bounded PCM retention with absolute positions and source-time anchors."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .stream import InputChunk, _validate_positive_integer

_SAMPLE_RATE = 16_000
_NANOSECONDS_PER_SECOND = 1_000_000_000
_SAMPLE_PERIOD_NS = _NANOSECONDS_PER_SECOND // _SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class SampleRange:
    """A half-open range in the processor-lifetime sample timeline."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Interpolated source metadata at one absolute sample position."""

    running_time_ns: int
    captured_at_ns: int
    generation: int
    discontinuity: bool


@dataclass(frozen=True, slots=True)
class _Anchor:
    position: int
    running_time_ns: int
    captured_at_ns: int
    generation: int
    discontinuity: bool


class AudioTimeline:
    """Own bounded PCM and source metadata for one processor-lifetime timeline."""

    def __init__(self, capacity: int, /) -> None:
        _validate_positive_integer("capacity", capacity)
        self._capacity = capacity
        self._samples = np.empty(capacity, dtype=np.dtype("<i2"))
        self._start = 0
        self._end = 0
        self._anchors: list[_Anchor] = []
        self._segment_timed: bool | None = None
        self._generation: int | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def retained_range(self) -> SampleRange:
        return SampleRange(self._start, self._end)

    @property
    def retained_samples(self) -> int:
        return self._end - self._start

    @property
    def next_position(self) -> int:
        return self._end

    def append(self, chunk: InputChunk, /) -> SampleRange:
        """Copy a complete chunk into storage and return its absolute range.

        Validation is performed before either PCM or anchors are changed, so a failed
        append leaves the retained timeline unchanged.
        """

        length = chunk.samples.size
        if length > self._capacity - self.retained_samples:
            raise BufferError("audio timeline capacity would be exceeded")

        starts_segment = self._generation is None or chunk.discontinuity or chunk.generation != self._generation
        segment_timed = chunk.running_time_ns != 0 if starts_segment else self._segment_timed
        if segment_timed is False and chunk.running_time_ns != 0:
            raise ValueError("running_time_ns cannot become available within an untimed segment")
        if segment_timed is True and chunk.running_time_ns == 0:
            raise ValueError("running_time_ns cannot become unavailable within a timed segment")
        if not starts_segment:
            self._validate_running_time(chunk.running_time_ns)

        start = self._end
        end = start + length
        discontinuity = starts_segment and self._generation is not None
        anchor = _Anchor(
            position=start,
            running_time_ns=chunk.running_time_ns,
            captured_at_ns=chunk.captured_at_ns,
            generation=chunk.generation,
            discontinuity=discontinuity,
        )

        first = start % self._capacity
        first_length = min(length, self._capacity - first)
        self._samples[first : first + first_length] = chunk.samples[:first_length]
        if first_length < length:
            self._samples[: length - first_length] = chunk.samples[first_length:]

        self._anchors.append(anchor)
        self._end = end
        self._generation = chunk.generation
        self._segment_timed = segment_timed
        return SampleRange(start, end)

    def read(self, start: int, end: int, /) -> NDArray[np.int16]:
        """Return an independent, writable copy of a retained nonempty range."""

        self._validate_range(start, end)
        length = end - start
        result = np.empty(length, dtype=np.dtype("<i2"))
        first = start % self._capacity
        first_length = min(length, self._capacity - first)
        result[:first_length] = self._samples[first : first + first_length]
        if first_length < length:
            result[first_length:] = self._samples[: length - first_length]
        return result

    def release_before(self, position: int, /) -> None:
        """Release retained samples below ``position`` while preserving timing lookup."""

        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("position must be an integer")
        if not self._start <= position <= self._end:
            raise ValueError("position must be within the retained range")

        self._start = position
        positions = [anchor.position for anchor in self._anchors]
        first_after = bisect_right(positions, position)
        keep_from = max(0, first_after - 1)
        self._anchors = self._anchors[keep_from:]

    def metadata_at(self, position: int, /) -> SourceMetadata:
        """Return interpolated source metadata for a retained sample position."""

        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("position must be an integer")
        if not self._start <= position < self._end:
            raise ValueError("position must be a retained sample position")
        positions = [anchor.position for anchor in self._anchors]
        anchor = self._anchors[bisect_right(positions, position) - 1]
        offset = position - anchor.position
        return SourceMetadata(
            running_time_ns=self._interpolate(anchor.running_time_ns, offset),
            captured_at_ns=self._interpolate(anchor.captured_at_ns, offset),
            generation=anchor.generation,
            discontinuity=anchor.discontinuity and position == anchor.position,
        )

    def _validate_running_time(self, running_time_ns: int) -> None:
        anchor = self._anchors[-1]
        expected = self._interpolate(anchor.running_time_ns, self._end - anchor.position)
        if abs(running_time_ns - expected) > _SAMPLE_PERIOD_NS:
            raise ValueError("running_time_ns disagrees with the sample timeline")

    @staticmethod
    def _interpolate(timestamp_ns: int, sample_offset: int) -> int:
        if timestamp_ns == 0:
            return 0
        return timestamp_ns + sample_offset * _NANOSECONDS_PER_SECOND // _SAMPLE_RATE

    def _validate_range(self, start: int, end: int) -> None:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise TypeError("end must be an integer")
        if start >= end:
            raise ValueError("range must be nonempty")
        if start < self._start or end > self._end:
            raise ValueError("range must be retained")
