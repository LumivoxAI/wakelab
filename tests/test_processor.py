from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamConfig,
    VadPolicyConfig,
    StreamStateError,
    WakePolicyConfig,
    WakeStreamProcessor,
)


class FakeBackend:
    sample_rate = 16_000
    frame_samples = 4

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self.frames: list[list[int]] = []
        self.reset_calls = 0
        self.close_calls = 0

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        self.frames.append(frame.tolist())
        return next(self._values)

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def config() -> StreamConfig:
    return StreamConfig(
        VadPolicyConfig(0.6, 0.4, 4, 4, 0, 0),
        WakePolicyConfig(0.7, 1, 4, 2),
        16,
    )


def chunk(values: list[int], **changes: object) -> InputChunk:
    fields: dict[str, object] = {
        "samples": np.asarray(values, dtype=np.dtype("<i2")),
        "running_time_ns": 0,
        "captured_at_ns": 0,
        "generation": 0,
        "discontinuity": False,
    }
    fields.update(changes)
    return InputChunk(**fields)  # type: ignore[arg-type]


def canonical(outputs: Sequence[OutputChunk]) -> list[tuple[int, bool, bool, int, bool]]:
    result: list[tuple[int, bool, bool, int, bool]] = []
    for output in outputs:
        samples = output.samples
        result.extend(
            (int(sample), output.is_speech, output.is_activated, output.generation, output.discontinuity and index == 0)
            for index, sample in enumerate(samples)
        )
    return result


def test_processor_conserves_audio_independent_of_input_partition() -> None:
    expected: list[tuple[int, bool, bool, int, bool]] | None = None
    for partition in ([12], [1, 2, 4, 5]):
        vad = FakeBackend([0.9, 0.9, 0.1])
        wake = FakeBackend([0.0, 0.0, 0.0])
        processor = WakeStreamProcessor(config(), vad, wake)
        outputs = []
        start = 0
        for length in partition:
            outputs.extend(processor.process(chunk(list(range(start, start + length)))))
            start += length
        outputs.extend(processor.finish())
        actual = canonical(outputs)
        if expected is None:
            expected = actual
        assert actual == expected
        assert [item[0] for item in actual] == list(range(12))


def test_processor_retroactively_activates_and_bypasses_wake_inference() -> None:
    vad = FakeBackend([0.9, 0.9, 0.9])
    wake = FakeBackend([0.0, 0.9])
    processor = WakeStreamProcessor(config(), vad, wake)

    outputs = processor.process(chunk(list(range(12)))) + processor.finish()

    assert canonical(outputs) == [
        (0, True, False, 0, False),
        (1, True, False, 0, False),
        (2, True, True, 0, False),
        (3, True, True, 0, False),
        (4, True, True, 0, False),
        (5, True, True, 0, False),
        (6, True, True, 0, False),
        (7, True, True, 0, False),
        (8, True, True, 0, False),
        (9, True, True, 0, False),
        (10, True, True, 0, False),
        (11, True, True, 0, False),
    ]
    assert wake.frames == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_processor_finalizes_hard_boundaries_and_propagates_discontinuity() -> None:
    vad = FakeBackend([0.9, 0.9])
    wake = FakeBackend([0.0, 0.0])
    processor = WakeStreamProcessor(config(), vad, wake)

    outputs = processor.process(chunk([0, 1, 2, 3]))
    outputs.extend(processor.process(chunk([4, 5, 6, 7], generation=1)))
    outputs.extend(processor.finish())

    assert canonical(outputs) == [
        (0, False, False, 0, False),
        (1, False, False, 0, False),
        (2, False, False, 0, False),
        (3, False, False, 0, False),
        (4, False, False, 1, True),
        (5, False, False, 1, False),
        (6, False, False, 1, False),
        (7, False, False, 1, False),
    ]
    assert vad.reset_calls == 2


def test_processor_close_is_idempotent_and_finish_is_terminal() -> None:
    vad = FakeBackend([0.1])
    wake = FakeBackend([])
    processor = WakeStreamProcessor(config(), vad, wake)

    outputs = processor.process(chunk([0, 1, 2, 3]))
    outputs.extend(processor.close())
    assert [item[0] for item in canonical(outputs)] == [0, 1, 2, 3]
    assert processor.close() == []
    assert (vad.close_calls, wake.close_calls) == (1, 1)
    with pytest.raises(StreamStateError):
        processor.process(chunk([4]))
