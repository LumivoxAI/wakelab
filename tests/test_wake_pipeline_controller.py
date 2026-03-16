from __future__ import annotations

import time
import wave
from typing import cast
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from lumivox_devicelab import CapturedChunk, CaptureHandler

import tools.wake_pipeline_support.controller as controller_module
from lumivox_wakelab import OutputChunk, StreamCloseError, WakeFrameDiagnostic, WakeStreamProcessor
from tools.wake_pipeline_support.charts import downsample_extrema
from tools.wake_pipeline_support.profile import BuiltProcessor, parse_profile, resolve_profile
from tools.wake_pipeline_support.controller import (
    DisplayEvent,
    SessionState,
    BoundedHandoff,
    DiagnosticHistory,
    SessionController,
    ActivatedWaveRecorder,
)

from .test_processor import FakeBackend, RecordingLogger, config
from .test_microphone_profile import profile_value


def captured(values: list[int], *, generation: int = 0, captured_at_ns: int = 0) -> CapturedChunk:
    return CapturedChunk(np.asarray(values, dtype=np.dtype("<i2")), generation, captured_at_ns, 0, False)


def test_handoff_has_exact_bounds_and_marks_survivor_after_drop() -> None:
    handoff = BoundedHandoff(2, 6)
    handoff.put_audio(captured([0, 1, 2]))
    handoff.put_audio(captured([3, 4, 5]))
    handoff.put_audio(captured([6, 7, 8]))

    first = handoff.get()
    second = handoff.get()

    assert first is not None and hasattr(first, "chunk")
    assert second is not None and hasattr(second, "chunk")
    assert first.chunk.samples.tolist() == [3, 4, 5]
    assert first.forced_discontinuity  # type: ignore[union-attr]
    assert second.chunk.samples.tolist() == [6, 7, 8]
    assert (handoff.dropped_chunks, handoff.dropped_samples) == (1, 3)


def test_handoff_never_drops_serialized_commands() -> None:
    handoff = BoundedHandoff(1, 3)
    handoff.put_command("rearm")
    handoff.put_audio(captured([0, 1, 2]))
    handoff.put_audio(captured([3, 4, 5]))

    command = handoff.get()
    audio = handoff.get()

    assert command is not None and getattr(command, "name") == "rearm"
    assert audio is not None and audio.forced_discontinuity  # type: ignore[union-attr]


def test_handoff_records_capture_latency_when_audio_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.wake_pipeline_support.controller.time.time_ns", lambda: 1_500_000)
    handoff = BoundedHandoff(1, 3)

    handoff.put_audio(captured([0, 1, 2], captured_at_ns=1_000_000))
    audio = handoff.get()

    assert audio is not None and hasattr(audio, "capture_latency_ms")
    assert audio.capture_latency_ms == 0.5


def test_downsampling_keeps_bucket_extrema_in_sample_order() -> None:
    points = ((0, 0.5), (1, 0.1), (2, 0.9), (4, 0.3), (5, 0.7))

    result = downsample_extrema(points, sample=lambda item: item[0], value=lambda item: item[1], bucket_samples=4)

    assert result == ((1, 0.1), (2, 0.9), (4, 0.3), (5, 0.7))


def test_diagnostic_history_records_only_nonzero_wake_streaks() -> None:
    history = DiagnosticHistory(16_000, 10)

    history.add_diagnostics(
        (
            WakeFrameDiagnostic(0, 0, 1_280, 0.6, False, 0),
            WakeFrameDiagnostic(0, 1_280, 2_560, 0.8, True, 1),
            WakeFrameDiagnostic(0, 2_560, 3_840, 0.9, True, 2),
        ),
        3_840,
    )

    assert tuple(history.events) == (
        DisplayEvent(2_560, "wake_hit", "score 0.8000, streak 1"),
        DisplayEvent(3_840, "wake_hit", "score 0.9000, streak 2"),
    )


def test_activation_recording_uses_finalized_output_and_is_readable_before_rearm(tmp_path: Path) -> None:
    path = tmp_path / "capture.wav"
    recorder = ActivatedWaveRecorder(path)
    recorder.begin_session(8)
    recorder.accept_output(_output([0, 1], activated=False))
    recorder.accept_output(_output([2, 3, 4, 5, 6, 7], activated=True))

    assert recorder.active
    assert _wave_values(path) == [2, 3, 4, 5, 6, 7]
    recorder.accept_output(_output([8, 9], activated=True))
    assert _wave_values(path) == list(range(2, 10))
    recorder.finish_cycle()


class FakePipeline:
    def __init__(self, handler: CaptureHandler) -> None:
        self.handler = handler
        self.stop_calls = 0

    def start(self, *, timeout: float = 10.0) -> None:
        self.handler.on_chunk(captured(list(range(8))))

    def stop(self, *, immediate: bool = False, timeout: float = 10.0) -> None:
        self.stop_calls += 1


def test_controller_processes_capture_and_shutdown_is_idempotent(tmp_path: Path) -> None:
    pipelines: list[FakePipeline] = []

    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        processor = WakeStreamProcessor(
            config(),
            FakeBackend([0.9, 0.9]),
            FakeBackend([0.0, 0.0]),
            logger=cast(RecordingLogger, logger),
            diagnostic_event_capacity=capacity,
        )
        return BuiltProcessor(processor, ("CPUExecutionProvider",))

    def pipeline_factory(handler: CaptureHandler, device_id: str, logger: object) -> FakePipeline:
        pipeline = FakePipeline(handler)
        pipelines.append(pipeline)
        return pipeline

    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=pipeline_factory,
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")

    controller.start("client", "microphone", profile)
    deadline = time.monotonic() + 1
    while controller.snapshot().current_sample != 8 and time.monotonic() < deadline:
        time.sleep(0.001)
    snapshot = controller.snapshot()

    assert snapshot.state is SessionState.LISTENING
    assert snapshot.current_sample == 8
    assert snapshot.providers == ("CPUExecutionProvider",)
    assert snapshot.diagnostics
    assert snapshot.vad_probability == 0.9
    assert snapshot.vad_is_speech
    assert snapshot.wake_state == "evaluating"
    with pytest.raises(RuntimeError, match="already active"):
        controller.start("other", "microphone", profile)
    controller.shutdown()
    controller.shutdown()
    assert controller.snapshot().state is SessionState.STOPPED
    assert pipelines[0].stop_calls == 2


def test_controller_rearm_completes_before_second_activation(tmp_path: Path) -> None:
    pipelines: list[FakePipeline] = []
    wake = FakeBackend([0.9, 0.9])

    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        processor = WakeStreamProcessor(
            config(),
            FakeBackend([0.9, 0.9, 0.9, 0.9]),
            wake,
            logger=cast(RecordingLogger, logger),
            diagnostic_event_capacity=capacity,
        )
        return BuiltProcessor(processor, ("CPUExecutionProvider",))

    def pipeline_factory(handler: CaptureHandler, device_id: str, logger: object) -> FakePipeline:
        pipeline = FakePipeline(handler)
        pipelines.append(pipeline)
        return pipeline

    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=pipeline_factory,
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")
    controller.start("client", "microphone", profile)

    deadline = time.monotonic() + 1
    while not controller.snapshot().activated and time.monotonic() < deadline:
        time.sleep(0.001)
    controller.rearm()
    assert not controller.snapshot().activated
    assert _wave_values(tmp_path / "capture.wav") == list(range(8))

    pipelines[0].handler.on_chunk(captured(list(range(8, 16))))
    deadline = time.monotonic() + 1
    while not controller.snapshot().activated and time.monotonic() < deadline:
        time.sleep(0.001)

    activations = [event for event in controller.snapshot().events if event.kind == "activation"]
    assert len(activations) == 2
    assert wake.frames == [[0, 1, 2, 3], [8, 9, 10, 11]]
    controller.rearm()
    assert _wave_values(tmp_path / "capture.wav") == list(range(8, 16))
    controller.shutdown()


def test_controller_recording_is_independent_of_disabled_diagnostics(tmp_path: Path) -> None:
    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        processor = WakeStreamProcessor(
            config(),
            FakeBackend([0.9, 0.9]),
            FakeBackend([0.0, 0.9]),
            logger=cast(RecordingLogger, logger),
            diagnostic_event_capacity=0,
        )
        return BuiltProcessor(processor, ("CPUExecutionProvider",))

    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=lambda handler, device_id, logger: FakePipeline(handler),
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")
    controller.start("client", "microphone", profile)

    deadline = time.monotonic() + 1
    while controller.snapshot().current_sample != 8 and time.monotonic() < deadline:
        time.sleep(0.001)

    snapshot = controller.snapshot()
    assert snapshot.diagnostics == ()
    assert not snapshot.activated
    controller.rearm()
    assert _wave_values(tmp_path / "capture.wav") == list(range(2, 8))
    controller.shutdown()


def test_controller_closes_processor_when_pipeline_construction_fails(tmp_path: Path) -> None:
    vad = FakeBackend([])
    wake = FakeBackend([])

    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        return BuiltProcessor(
            WakeStreamProcessor(
                config(),
                vad,
                wake,
                logger=cast(RecordingLogger, logger),
                diagnostic_event_capacity=capacity,
            ),
            (),
        )

    def fail_pipeline(handler: CaptureHandler, device_id: str, logger: object) -> FakePipeline:
        raise RuntimeError("pipeline construction failed")

    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=fail_pipeline,
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")

    with pytest.raises(RuntimeError, match="pipeline construction failed"):
        controller.start("client", "microphone", profile)

    assert controller.snapshot().state is SessionState.ERROR
    assert (vad.close_calls, wake.close_calls) == (1, 1)


class CloseOutputProcessor(WakeStreamProcessor):
    def close(self) -> list[OutputChunk]:
        super().close()
        output = OutputChunk(np.asarray([99], dtype=np.dtype("<i2")), 0, 0, 0, False, True, True)
        raise StreamCloseError("injected close failure", [output], (RuntimeError("resource failure"),))


def test_controller_records_outputs_carried_by_close_error(tmp_path: Path) -> None:
    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        processor = CloseOutputProcessor(
            config(),
            FakeBackend([0.1, 0.1]),
            FakeBackend([]),
            logger=cast(RecordingLogger, logger),
        )
        return BuiltProcessor(processor, ())

    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=lambda handler, device_id, logger: FakePipeline(handler),
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")
    controller.start("client", "microphone", profile)
    controller.shutdown()

    assert controller.snapshot().state is SessionState.ERROR
    assert _wave_values(tmp_path / "capture.wav") == [99]


class BlockingBackend(FakeBackend):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__([0.1, 0.1])
        self._started = started
        self._release = release

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        self._started.set()
        self._release.wait(timeout=1)
        return super().infer(frame)


def test_controller_remains_non_restartable_until_timed_out_worker_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    def builder(profile: object, logger: object, capacity: int) -> BuiltProcessor:
        return BuiltProcessor(
            WakeStreamProcessor(
                config(),
                BlockingBackend(started, release),
                FakeBackend([]),
                logger=cast(RecordingLogger, logger),
            ),
            (),
        )

    monkeypatch.setattr(controller_module, "_WORKER_JOIN_TIMEOUT_SECONDS", 0)
    controller = SessionController(
        RecordingLogger(),
        recording_path=tmp_path / "capture.wav",
        processor_builder=builder,
        pipeline_factory=lambda handler, device_id, logger: FakePipeline(handler),
    )
    profile = resolve_profile(parse_profile(profile_value()), tmp_path / "profile.json")
    controller.start("client", "microphone", profile)
    assert started.wait(timeout=1)

    controller.stop()
    assert controller.snapshot().state is SessionState.STOPPING
    with pytest.raises(RuntimeError, match="already active"):
        controller.start("other", "microphone", profile)

    release.set()
    deadline = time.monotonic() + 1
    while controller.snapshot().state is not SessionState.STOPPED and time.monotonic() < deadline:
        time.sleep(0.001)
    assert controller.snapshot().state is SessionState.STOPPED


class FakeSpeaker:
    def __init__(self) -> None:
        self.submitted: list[int] = []
        self.stop_immediate: bool | None = None

    def start(self, *, timeout: float = 10.0) -> None:
        pass

    def submit(self, data: np.ndarray) -> None:
        self.submitted = data.tolist()

    def stop(self, *, immediate: bool = False, timeout: float = 10.0) -> None:
        self.stop_immediate = immediate


def test_controller_plays_finalized_activation_recording(tmp_path: Path) -> None:
    path = tmp_path / "capture.wav"
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(np.asarray([1, 2, 3], dtype="<i2").tobytes())
    speakers: list[FakeSpeaker] = []

    def speaker_factory(device_id: str, logger: object) -> FakeSpeaker:
        speaker = FakeSpeaker()
        speakers.append(speaker)
        return speaker

    controller = SessionController(
        RecordingLogger(),
        recording_path=path,
        speaker_factory=speaker_factory,
    )

    assert controller.play("speaker")
    assert speakers[0].submitted == [1, 2, 3]
    assert speakers[0].stop_immediate is False
    assert not controller.snapshot().playback_active


def _wave_values(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as recording:
        return cast(list[int], np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").tolist())


def _output(values: list[int], *, activated: bool) -> OutputChunk:
    return OutputChunk(np.asarray(values, dtype="<i2"), 0, 0, 0, False, True, activated)
