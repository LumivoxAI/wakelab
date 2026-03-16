from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from lumivox_wakelab.stream import InputChunk
from lumivox_wakelab._framing import VadFramer, FrameResult, WakeWordFramer
from lumivox_wakelab._timeline import SampleRange, AudioTimeline


class FakeBackend:
    def __init__(self, frame_samples: int, *, failure_calls: Iterable[int] = (), result_scale: float = 1.0) -> None:
        self.sample_rate = 16_000
        self.frame_samples = frame_samples
        self.frames: list[np.ndarray[tuple[int], np.dtype[np.int16]]] = []
        self.reset_calls = 0
        self._failure_calls = set(failure_calls)
        self._result_scale = result_scale

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        call = len(self.frames)
        if call in self._failure_calls:
            self._failure_calls.remove(call)
            raise RuntimeError("injected inference failure")
        self.frames.append(frame.copy())
        return float(frame[0]) * self._result_scale

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        pass


def append(timeline: AudioTimeline, values: list[int]) -> SampleRange:
    return timeline.append(
        InputChunk(
            samples=np.asarray(values, dtype=np.dtype("<i2")),
            running_time_ns=0,
            captured_at_ns=0,
            generation=0,
            discontinuity=False,
        )
    )


def test_vad_framing_is_independent_of_input_partition() -> None:
    expected = [
        FrameResult(SampleRange(0, 4), 0.0),
        FrameResult(SampleRange(4, 8), 4.0),
        FrameResult(SampleRange(8, 12), 8.0),
    ]

    for partitions in ([12], [1, 2, 4, 5]):
        timeline = AudioTimeline(16)
        backend = FakeBackend(4)
        framer = VadFramer(backend)
        framer.start_segment(0)
        results: list[FrameResult] = []
        start = 0
        for length in partitions:
            append(timeline, list(range(start, start + length)))
            start += length
            results.extend(framer.consume(timeline, timeline.next_position))

        assert results == expected
        assert [frame.tolist() for frame in backend.frames] == [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
        ]


def test_vad_keeps_incomplete_tail_and_restarts_at_boundary() -> None:
    timeline = AudioTimeline(16)
    backend = FakeBackend(4)
    framer = VadFramer(backend)
    framer.start_segment(0)
    append(timeline, [0, 1, 2, 3, 4, 5])

    assert framer.consume(timeline, 6) == [FrameResult(SampleRange(0, 4), 0.0)]
    framer.start_segment(6)
    append(timeline, [6, 7, 8, 9])

    assert framer.consume(timeline, 10) == [FrameResult(SampleRange(6, 10), 6.0)]
    assert [frame.tolist() for frame in backend.frames] == [[0, 1, 2, 3], [6, 7, 8, 9]]
    assert backend.reset_calls == 2


def test_failed_vad_frame_does_not_advance_cursor() -> None:
    timeline = AudioTimeline(8)
    append(timeline, list(range(8)))
    backend = FakeBackend(4, failure_calls=[1])
    framer = VadFramer(backend)
    framer.start_segment(0)

    with pytest.raises(RuntimeError, match="injected"):
        framer.consume(timeline, 8)

    assert framer.position == 4
    assert framer.consume(timeline, 8) == [FrameResult(SampleRange(4, 8), 4.0)]


def test_wake_framing_starts_at_arbitrary_candidate_position_and_ends_tail() -> None:
    timeline = AudioTimeline(16)
    append(timeline, list(range(12)))
    backend = FakeBackend(4, result_scale=0.125)
    framer = WakeWordFramer(backend)
    framer.start_candidate(3)

    assert framer.consume(timeline, 12) == [
        FrameResult(SampleRange(3, 7), 0.375),
        FrameResult(SampleRange(7, 11), 0.875),
    ]
    framer.end_candidate(12)

    assert not framer.evaluating
    assert [frame.tolist() for frame in backend.frames] == [[3, 4, 5, 6], [7, 8, 9, 10]]


def test_wake_framing_advances_only_after_each_successful_frame() -> None:
    timeline = AudioTimeline(8)
    append(timeline, list(range(8)))
    backend = FakeBackend(4, failure_calls=[1])
    framer = WakeWordFramer(backend)
    framer.start_candidate(0)

    assert framer.consume_next(timeline, 8) == FrameResult(SampleRange(0, 4), 0.0)
    with pytest.raises(RuntimeError, match="injected"):
        framer.consume_next(timeline, 8)
    assert framer.position == 4


def test_wake_candidate_rejects_overlap_and_reset_cancels_it() -> None:
    timeline = AudioTimeline(8)
    append(timeline, list(range(8)))
    backend = FakeBackend(4)
    framer = WakeWordFramer(backend)
    framer.start_candidate(0)
    framer.consume(timeline, 4)

    with pytest.raises(ValueError, match="precedes"):
        framer.end_candidate(3)
    with pytest.raises(RuntimeError, match="already active"):
        framer.start_candidate(4)

    framer.reset()
    assert not framer.evaluating
    framer.start_candidate(4)
    assert backend.reset_calls == 3


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_wake_framer_rejects_invalid_backend_probabilities(value: float) -> None:
    timeline = AudioTimeline(4)
    append(timeline, [1, 2, 3, 4])
    backend = FakeBackend(4, result_scale=value)
    framer = WakeWordFramer(backend)
    framer.start_candidate(0)

    with pytest.raises(ValueError, match="finite scalar|probability"):
        framer.consume_next(timeline, 4)
    assert framer.position == 0


@pytest.mark.parametrize("sample_rate,frame_samples", [(8_000, 4), (16_000, 0), (16_000, True)])
def test_framers_reject_incompatible_backend_capabilities(sample_rate: int, frame_samples: int) -> None:
    backend = FakeBackend(4)
    backend.sample_rate = sample_rate
    backend.frame_samples = frame_samples

    with pytest.raises(ValueError):
        VadFramer(backend)
