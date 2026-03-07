from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from lumivox_devicelab import AudioDevice, CapturedChunk, DeviceSnapshot

from lumivox_wakelab import InputChunk, OutputChunk, DiagnosticDrain
from examples.microphone_wake_recorder import (
    SAMPLE_RATE,
    DEFAULT_PROFILE,
    ActivatedAudioHandler,
    write_wave,
    load_profile,
    select_microphone,
)


class FakeProcessor:
    def __init__(self, outputs: list[OutputChunk]) -> None:
        self._outputs = outputs
        self.inputs: list[InputChunk] = []

    def process(self, chunk: InputChunk, /) -> list[OutputChunk]:
        self.inputs.append(chunk)
        outputs = self._outputs
        self._outputs = []
        return outputs

    def rearm(self) -> list[OutputChunk]:
        return []

    def discontinue(self) -> list[OutputChunk]:
        return []

    def finish(self) -> list[OutputChunk]:
        return []

    def drain_failed(self) -> list[OutputChunk]:
        return []

    def drain_diagnostics(self) -> DiagnosticDrain:
        return DiagnosticDrain((), 0)

    def close(self) -> list[OutputChunk]:
        return []


def output(values: list[int], *, activated: bool) -> OutputChunk:
    return OutputChunk(
        samples=np.asarray(values, dtype=np.dtype("<i2")),
        running_time_ns=0,
        captured_at_ns=0,
        generation=0,
        discontinuity=False,
        is_speech=True,
        is_activated=activated,
    )


def device(identifier: str, *, default: bool = False) -> AudioDevice:
    return AudioDevice(identifier, f"Microphone {identifier}", default, ())


def test_recommended_profile_is_valid_and_resolves_paths() -> None:
    profile = load_profile(DEFAULT_PROFILE)

    assert profile.name == "marina-local-demo-v1"
    assert profile.model_root == DEFAULT_PROFILE.parent.parent / "data"
    assert profile.classifier_path == DEFAULT_PROFILE.parent.parent / "data/oww/marina/marina_dnn_ex1000_64.onnx"
    assert profile.stream_config.wake_policy.pre_roll_samples == 24_000


def test_profile_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_version": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 1"):
        load_profile(path)


def test_select_microphone_uses_default_or_exact_id() -> None:
    first = device("first")
    default = device("default", default=True)
    snapshot = DeviceSnapshot((first, default), default)

    assert select_microphone(snapshot, None) is default
    assert select_microphone(snapshot, "first") is first
    with pytest.raises(ValueError, match="microphone ID not found"):
        select_microphone(snapshot, "missing")


def test_select_microphone_requires_reported_default() -> None:
    snapshot = DeviceSnapshot((device("only"),), None)

    with pytest.raises(ValueError, match="no default microphone"):
        select_microphone(snapshot, None)


def test_handler_maps_capture_metadata_and_keeps_exact_target() -> None:
    processor = FakeProcessor(
        [
            output([1, 2], activated=False),
            output([3, 4, 5], activated=True),
            output([6, 7, 8], activated=True),
        ]
    )
    statuses: list[str] = []
    handler = ActivatedAudioHandler(processor, 5, status=statuses.append)
    captured = CapturedChunk(
        np.asarray([10, 11], dtype=np.dtype("<i2")),
        generation=3,
        captured_at_ns=20,
        running_time_ns=10,
        discontinuity=True,
    )

    handler.on_chunk(captured)

    assert handler.complete.is_set()
    assert handler.recorded_samples == 5
    assert handler.samples().tolist() == [3, 4, 5, 6, 7]
    assert len(statuses) == 2
    assert len(processor.inputs) == 1
    actual = processor.inputs[0]
    assert actual.samples.tolist() == [10, 11]
    assert (actual.generation, actual.captured_at_ns, actual.running_time_ns, actual.discontinuity) == (3, 20, 10, True)


def test_write_wave_writes_pcm_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "recording.wav"
    samples = np.asarray([-2, 0, 2], dtype=np.dtype("<i2"))

    write_wave(path, samples)

    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, SAMPLE_RATE, 3)
        assert wav.readframes(3) == samples.tobytes()
    with pytest.raises(FileExistsError):
        write_wave(path, samples)
