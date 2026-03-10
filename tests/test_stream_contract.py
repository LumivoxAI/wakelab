from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import lumivox_wakelab
from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamError,
    StreamConfig,
    StreamProcessor,
    VadPolicyConfig,
    StreamCloseError,
    StreamStateError,
    WakePolicyConfig,
    StreamProcessingError,
)


def pcm(length: int = 4) -> np.ndarray[tuple[int], np.dtype[np.int16]]:
    return np.arange(length, dtype=np.dtype("<i2"))


def input_chunk(**changes: object) -> InputChunk:
    values: dict[str, object] = {
        "samples": pcm(),
        "running_time_ns": 10,
        "captured_at_ns": 20,
        "generation": 1,
        "discontinuity": False,
    }
    values.update(changes)
    return InputChunk(**values)  # type: ignore[arg-type]


def output_chunk(**changes: object) -> OutputChunk:
    values: dict[str, object] = {
        "samples": pcm(),
        "running_time_ns": 10,
        "captured_at_ns": 20,
        "generation": 1,
        "discontinuity": False,
        "is_speech": True,
        "is_activated": False,
    }
    values.update(changes)
    return OutputChunk(**values)  # type: ignore[arg-type]


def vad_config(**changes: object) -> VadPolicyConfig:
    values: dict[str, object] = {
        "speech_threshold": 0.6,
        "silence_threshold": 0.4,
        "minimum_speech_samples": 512,
        "minimum_silence_samples": 1_024,
        "left_padding_samples": 160,
        "right_padding_samples": 320,
    }
    values.update(changes)
    return VadPolicyConfig(**values)  # type: ignore[arg-type]


def wake_config(**changes: object) -> WakePolicyConfig:
    values: dict[str, object] = {
        "score_threshold": 0.7,
        "consecutive_score_count": 2,
        "silence_bridge_samples": 1_280,
        "pre_roll_samples": 4_800,
    }
    values.update(changes)
    return WakePolicyConfig(**values)  # type: ignore[arg-type]


def test_chunks_have_frozen_metadata_and_mutable_samples() -> None:
    chunk = output_chunk()

    with pytest.raises(FrozenInstanceError):
        chunk.is_speech = False  # type: ignore[misc]

    chunk.samples[0] = 42
    assert chunk.samples[0] == 42
    assert not hasattr(chunk, "__dict__")


@pytest.mark.parametrize("factory", [input_chunk, output_chunk])
@pytest.mark.parametrize(
    ("samples", "error_type"),
    [
        ([1, 2], TypeError),
        (np.zeros(2, dtype=np.float32), TypeError),
        (np.zeros(2, dtype=np.dtype(">i2")), TypeError),
        (np.zeros((1, 2), dtype=np.dtype("<i2")), ValueError),
        (np.zeros(0, dtype=np.dtype("<i2")), ValueError),
    ],
)
def test_chunks_reject_invalid_sample_arrays(factory: object, samples: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        factory(samples=samples)  # type: ignore[operator]


@pytest.mark.parametrize("factory", [input_chunk, output_chunk])
def test_chunks_reject_read_only_samples(factory: object) -> None:
    samples = pcm()
    samples.flags.writeable = False

    with pytest.raises(ValueError, match="writable"):
        factory(samples=samples)  # type: ignore[operator]


@pytest.mark.parametrize("factory", [input_chunk, output_chunk])
@pytest.mark.parametrize("field", ["running_time_ns", "captured_at_ns", "generation"])
@pytest.mark.parametrize(("value", "error_type"), [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_chunks_reject_invalid_integer_metadata(
    factory: object,
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        factory(**{field: value})  # type: ignore[operator]


@pytest.mark.parametrize("factory", [input_chunk, output_chunk])
def test_chunks_require_boolean_discontinuity(factory: object) -> None:
    with pytest.raises(TypeError):
        factory(discontinuity=0)  # type: ignore[operator]


@pytest.mark.parametrize("field", ["is_speech", "is_activated"])
def test_output_requires_boolean_classification(field: str) -> None:
    with pytest.raises(TypeError):
        output_chunk(**{field: 1})


@pytest.mark.parametrize("field", ["speech_threshold", "silence_threshold"])
@pytest.mark.parametrize("value", [True, "0.5"])
def test_vad_thresholds_require_real_numbers(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        vad_config(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, -math.inf, math.nan])
def test_vad_thresholds_must_be_finite_probabilities(value: float) -> None:
    with pytest.raises(ValueError):
        vad_config(speech_threshold=value)


@pytest.mark.parametrize(("silence", "speech"), [(0.5, 0.5), (0.6, 0.5)])
def test_vad_thresholds_require_ordered_hysteresis(silence: float, speech: float) -> None:
    with pytest.raises(ValueError):
        vad_config(silence_threshold=silence, speech_threshold=speech)


@pytest.mark.parametrize("field", ["minimum_speech_samples", "minimum_silence_samples"])
@pytest.mark.parametrize(
    ("value", "error_type"), [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)]
)
def test_vad_minimums_require_positive_integers(field: str, value: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        vad_config(**{field: value})


@pytest.mark.parametrize("field", ["left_padding_samples", "right_padding_samples"])
@pytest.mark.parametrize(("value", "error_type"), [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_vad_padding_requires_nonnegative_integers(field: str, value: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        vad_config(**{field: value})


@pytest.mark.parametrize("value", [True, "0.5"])
def test_wake_threshold_requires_a_real_number(value: object) -> None:
    with pytest.raises(TypeError):
        wake_config(score_threshold=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, -math.inf, math.nan])
def test_wake_threshold_must_be_a_finite_probability(value: float) -> None:
    with pytest.raises(ValueError):
        wake_config(score_threshold=value)


@pytest.mark.parametrize(
    ("value", "error_type"), [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)]
)
def test_wake_consecutive_count_requires_a_positive_integer(value: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        wake_config(consecutive_score_count=value)


@pytest.mark.parametrize("field", ["silence_bridge_samples", "pre_roll_samples"])
@pytest.mark.parametrize(("value", "error_type"), [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_wake_durations_require_nonnegative_integers(field: str, value: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        wake_config(**{field: value})


def test_stream_config_is_nested_and_has_no_defaults() -> None:
    config = StreamConfig(vad_config(), wake_config(), 16_000)

    assert config.vad_policy.minimum_speech_samples == 512
    assert config.wake_policy.consecutive_score_count == 2
    with pytest.raises(TypeError):
        StreamConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"vad_policy": object()}, TypeError),
        ({"wake_policy": object()}, TypeError),
        ({"max_retained_audio_samples": True}, TypeError),
        ({"max_retained_audio_samples": 1.5}, TypeError),
        ({"max_retained_audio_samples": 0}, ValueError),
        ({"max_retained_audio_samples": -1}, ValueError),
    ],
)
def test_stream_config_rejects_invalid_values(changes: dict[str, object], error_type: type[Exception]) -> None:
    values: dict[str, object] = {
        "vad_policy": vad_config(),
        "wake_policy": wake_config(),
        "max_retained_audio_samples": 16_000,
    }
    values.update(changes)
    with pytest.raises(error_type):
        StreamConfig(**values)  # type: ignore[arg-type]


def test_stream_processor_protocol_has_the_public_operation_shape() -> None:
    assert set(StreamProcessor.__dict__) >= {
        "process",
        "rearm",
        "discontinue",
        "finish",
        "drain_failed",
        "drain_diagnostics",
        "close",
    }


def test_stream_error_hierarchy_and_close_payload() -> None:
    outputs = [output_chunk()]
    resource_errors = (RuntimeError("vad"), RuntimeError("wake"))
    error = StreamCloseError("resource closure failed", outputs, resource_errors)

    assert isinstance(error, StreamError)
    assert issubclass(StreamStateError, StreamError)
    assert issubclass(StreamProcessingError, StreamError)
    assert error.outputs is outputs
    assert error.errors is resource_errors


def test_top_level_exports_are_intentional() -> None:
    assert lumivox_wakelab.__all__ == [
        "ActivationDiagnostic",
        "ContinuityBoundaryDiagnostic",
        "ContinuityBoundaryReason",
        "DiagnosticDrain",
        "DiagnosticEvent",
        "DiagnosticFailureOperation",
        "FailureDiagnostic",
        "InputChunk",
        "OutputChunk",
        "RearmDiagnostic",
        "StreamCloseError",
        "StreamConfig",
        "StreamError",
        "StreamProcessingError",
        "StreamProcessor",
        "StreamStateError",
        "VadPolicyConfig",
        "VadFrameDiagnostic",
        "VadSpeechDiagnostic",
        "VadTransitionReason",
        "WakeCandidateEndedDiagnostic",
        "WakeCandidateEndReason",
        "WakeCandidateStartedDiagnostic",
        "WakeFrameDiagnostic",
        "WakePolicyConfig",
        "WakeStreamProcessor",
    ]
