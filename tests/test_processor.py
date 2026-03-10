from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamConfig,
    VadPolicyConfig,
    StreamCloseError,
    StreamStateError,
    WakePolicyConfig,
    WakeStreamProcessor,
    ActivationDiagnostic,
    StreamProcessingError,
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


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def bind(self, **values: object) -> RecordingLogger:
        return self

    def debug(self, event: str, **values: object) -> None:
        pass

    def info(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def warning(self, event: str, **values: object) -> None:
        pass

    def error(self, event: str, **values: object) -> None:
        self.events.append((event, values))

    def critical(self, event: str, **values: object) -> None:
        pass

    def exception(self, event: str, **values: object) -> None:
        pass


class FailingBackend(FakeBackend):
    def __init__(self, values: list[float], fail_at: int, close_error: Exception | None = None) -> None:
        super().__init__(values)
        self._fail_at = fail_at
        self._calls = 0
        self._close_error = close_error

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        self._calls += 1
        if self._calls == self._fail_at:
            raise RuntimeError("injected inference failure")
        return super().infer(frame)

    def close(self) -> None:
        super().close()
        if self._close_error is not None:
            raise self._close_error


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
        processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())
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
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())

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


def test_processor_can_activate_again_after_rearm_in_same_segment() -> None:
    vad = FakeBackend([0.9, 0.9, 0.9, 0.9])
    wake = FakeBackend([0.9, 0.9])
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger(), diagnostic_event_capacity=64)

    first = processor.process(chunk(list(range(8))))
    rearmed = processor.rearm()
    second = processor.process(chunk(list(range(8, 16))))
    outputs = first + rearmed + second + processor.finish()

    assert wake.frames == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert wake.reset_calls == 4
    activations = [item for item in processor.drain_diagnostics().events if isinstance(item, ActivationDiagnostic)]
    assert [(item.activation_start_sample, item.confirming_end_sample) for item in activations] == [(0, 4), (8, 12)]
    assert [activated for sample, _, activated, _, _ in canonical(outputs) if sample in {2, 10}] == [True, True]


def test_processor_finalizes_hard_boundaries_and_propagates_discontinuity() -> None:
    vad = FakeBackend([0.9, 0.9])
    wake = FakeBackend([0.0, 0.0])
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())

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


def test_processor_input_discontinuity_resets_without_crossing_frames() -> None:
    vad = FakeBackend([0.1])
    wake = FakeBackend([])
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())

    outputs = processor.process(chunk([0, 1]))
    outputs.extend(processor.process(chunk([2, 3, 4, 5], discontinuity=True)))
    outputs.extend(processor.finish())

    assert [item[0] for item in canonical(outputs)] == list(range(6))
    assert vad.frames == [[2, 3, 4, 5]]
    assert vad.reset_calls == 2
    assert [(output.generation, output.discontinuity) for output in outputs] == [(0, False), (0, True)]


def test_processor_retention_validation_covers_left_padding_and_silence_bridge() -> None:
    left_padding = StreamConfig(
        VadPolicyConfig(0.6, 0.4, 4, 4, 8, 0),
        WakePolicyConfig(0.7, 1, 0, 0),
        16,
    )
    bridge = StreamConfig(
        VadPolicyConfig(0.6, 0.4, 4, 4, 0, 0),
        WakePolicyConfig(0.7, 1, 10, 0),
        16,
    )

    with pytest.raises(ValueError, match="at least 16"):
        WakeStreamProcessor(
            StreamConfig(left_padding.vad_policy, left_padding.wake_policy, 15),
            FakeBackend([]),
            FakeBackend([]),
            logger=RecordingLogger(),
        )
    with pytest.raises(ValueError, match="at least 16"):
        WakeStreamProcessor(
            StreamConfig(bridge.vad_policy, bridge.wake_policy, 15),
            FakeBackend([]),
            FakeBackend([]),
            logger=RecordingLogger(),
        )

    left_vad = FakeBackend([0.1, 0.1, 0.1, 0.9, 0.9])
    left_processor = WakeStreamProcessor(left_padding, left_vad, FakeBackend([0.0] * 5), logger=RecordingLogger())
    left_outputs = left_processor.process(chunk(list(range(20)))) + left_processor.finish()
    assert [item[0] for item in canonical(left_outputs)] == list(range(20))

    bridge_vad = FakeBackend([0.9, 0.9, 0.1, 0.1, 0.1])
    bridge_processor = WakeStreamProcessor(bridge, bridge_vad, FakeBackend([0.0, 0.0]), logger=RecordingLogger())
    bridge_outputs = bridge_processor.process(chunk(list(range(20)))) + bridge_processor.finish()
    assert [item[0] for item in canonical(bridge_outputs)] == list(range(20))


def test_processor_close_is_idempotent_and_finish_is_terminal() -> None:
    vad = FakeBackend([0.1])
    wake = FakeBackend([])
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())

    outputs = processor.process(chunk([0, 1, 2, 3]))
    outputs.extend(processor.close())
    assert [item[0] for item in canonical(outputs)] == [0, 1, 2, 3]
    assert processor.close() == []
    assert (vad.close_calls, wake.close_calls) == (1, 1)
    with pytest.raises(StreamStateError):
        processor.process(chunk([4]))


def test_processor_finish_is_idempotent() -> None:
    processor = WakeStreamProcessor(config(), FakeBackend([0.1]), FakeBackend([]), logger=RecordingLogger())

    outputs = processor.process(chunk([0, 1, 2, 3]))
    outputs.extend(processor.finish())

    assert [item[0] for item in canonical(outputs)] == [0, 1, 2, 3]
    assert processor.finish() == []
    with pytest.raises(StreamStateError):
        processor.process(chunk([4]))


def test_processor_failure_accepts_and_drains_the_complete_input_without_further_inference() -> None:
    logger = RecordingLogger()
    vad = FailingBackend([0.9], fail_at=2)
    wake = FakeBackend([])
    processor = WakeStreamProcessor(config(), vad, wake, logger=logger)

    with pytest.raises(StreamProcessingError):
        processor.process(chunk(list(range(12))))

    assert processor.accepted_samples == 12
    with pytest.raises(StreamStateError):
        processor.process(chunk([12]))
    outputs = processor.drain_failed()
    assert [item[0] for item in canonical(outputs)] == list(range(12))
    assert processor.drain_failed() == []
    assert vad.frames == [[0, 1, 2, 3]]
    assert any(event == "stream_processing_failed" for event, _ in logger.events)


def test_processor_failure_drain_preserves_committed_speech_without_wake_inference() -> None:
    vad = FakeBackend([0.9, 0.9, 0.9])
    wake = FailingBackend([], fail_at=1)
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())

    with pytest.raises(StreamProcessingError):
        processor.process(chunk(list(range(12))))

    outputs = processor.drain_failed()
    assert [(sample, speech, activated) for sample, speech, activated, _, _ in canonical(outputs)] == [
        (sample, True, False) for sample in range(12)
    ]
    assert wake.frames == []


def test_processor_close_aggregates_resource_errors_after_draining() -> None:
    vad = FailingBackend([0.1], fail_at=99, close_error=RuntimeError("vad close"))
    wake = FailingBackend([], fail_at=99, close_error=RuntimeError("wake close"))
    processor = WakeStreamProcessor(config(), vad, wake, logger=RecordingLogger())
    processor.process(chunk([0, 1]))

    with pytest.raises(StreamCloseError) as caught:
        processor.close()

    assert [item[0] for item in canonical(caught.value.outputs)] == [0, 1]
    assert [str(error) for error in caught.value.errors] == ["vad close", "wake close"]
    assert (vad.close_calls, wake.close_calls) == (1, 1)
    assert processor.close() == []
