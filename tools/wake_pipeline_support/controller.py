"""NiceGUI-independent capture controller and bounded diagnostic data model."""

from __future__ import annotations

import time
import wave
from enum import StrEnum
from typing import BinaryIO, Protocol
from pathlib import Path
from threading import Lock, Event, Thread, Condition, current_thread
from contextlib import suppress
from collections import deque
from dataclasses import replace, dataclass
from collections.abc import Callable

import numpy as np
from lumivox_devicelab import (
    AudioFormat,
    CapturedChunk,
    CaptureHandler,
    PlaybackSubmissionError,
    SpeakerPlaybackPipeline,
    MicrophoneCapturePipeline,
)
from lumivox_core.logger import Logger

from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    DiagnosticEvent,
    RearmDiagnostic,
    StreamCloseError,
    FailureDiagnostic,
    VadFrameDiagnostic,
    VadSpeechDiagnostic,
    WakeFrameDiagnostic,
    WakeStreamProcessor,
    ActivationDiagnostic,
    ContinuityBoundaryDiagnostic,
    WakeCandidateEndedDiagnostic,
    WakeCandidateStartedDiagnostic,
)

from .profile import BuiltProcessor, MicrophoneProfile, build_processor

SAMPLE_RATE = 16_000
DEFAULT_HISTORY_SAMPLES = 60 * SAMPLE_RATE
DEFAULT_DISPLAY_BUCKET_SAMPLES = SAMPLE_RATE // 5
DEFAULT_EVENT_CAPACITY = 10_000
LATENCY_AVERAGE_WINDOW_NS = 500_000_000
DEFAULT_RECORDING_PATH = Path(__file__).parents[2] / "wake-pipeline-capture.wav"
_WORKER_JOIN_TIMEOUT_SECONDS = 10


class SessionState(StrEnum):
    READY = "ready"
    LOADING = "loading"
    LISTENING = "listening"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DisplayEvent:
    sample: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    state: SessionState
    owner_id: str | None
    profile_name: str | None
    generation: int | None
    current_sample: int
    retained_samples: int
    queue_depth: int
    queue_samples: int
    dropped_chunks: int
    dropped_samples: int
    diagnostic_drops: int
    capture_latency_ms: float | None
    processing_latency_ms: float | None
    vad_probability: float | None
    vad_is_speech: bool
    wake_score: float | None
    wake_hit_count: int
    wake_state: str
    activated: bool
    recording_path: str
    recording_samples: int
    recording_active: bool
    recording_available: bool
    playback_active: bool
    providers: tuple[str, ...]
    diagnostics: tuple[DiagnosticEvent, ...]
    events: tuple[DisplayEvent, ...]
    error: str | None

    @property
    def active(self) -> bool:
        return self.state in {SessionState.LOADING, SessionState.LISTENING, SessionState.STOPPING}


@dataclass(frozen=True, slots=True)
class _AudioItem:
    chunk: CapturedChunk
    forced_discontinuity: bool = False
    capture_latency_ms: float = 0.0


@dataclass(slots=True)
class _Command:
    name: str
    completed: Event
    error: BaseException | None = None


type _HandoffItem = _AudioItem | _Command


class BoundedHandoff:
    """FIFO handoff that drops old audio but never serialized commands."""

    def __init__(self, max_chunks: int, max_samples: int, /) -> None:
        if max_chunks <= 0 or max_samples <= 0:
            raise ValueError("handoff bounds must be positive")
        self._max_chunks = max_chunks
        self._max_samples = max_samples
        self._items: deque[_HandoffItem] = deque()
        self._audio_chunks = 0
        self._samples = 0
        self._pending_gap = False
        self._closed = False
        self._condition = Condition()
        self.dropped_chunks = 0
        self.dropped_samples = 0

    @property
    def depth(self) -> int:
        with self._condition:
            return self._audio_chunks

    @property
    def samples(self) -> int:
        with self._condition:
            return self._samples

    def put_audio(self, chunk: CapturedChunk) -> None:
        capture_latency_ms = (time.time_ns() - chunk.captured_at_ns) / 1_000_000
        with self._condition:
            if self._closed:
                return
            size = int(chunk.samples.size)
            if size > self._max_samples:
                self.dropped_chunks += 1
                self.dropped_samples += size
                self._pending_gap = True
                return
            while self._audio_chunks >= self._max_chunks or self._samples + size > self._max_samples:
                if not self._drop_oldest_audio():
                    break
            item = _AudioItem(chunk, self._pending_gap, capture_latency_ms)
            self._pending_gap = False
            self._items.append(item)
            self._audio_chunks += 1
            self._samples += size
            self._condition.notify()

    def put_command(self, name: str) -> _Command | None:
        with self._condition:
            if self._closed:
                return None
            command = _Command(name, Event())
            self._items.append(command)
            self._condition.notify()
            return command

    def get(self) -> _HandoffItem | None:
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if not self._items:
                return None
            item = self._items.popleft()
            if isinstance(item, _AudioItem):
                self._audio_chunks -= 1
                self._samples -= int(item.chunk.samples.size)
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _drop_oldest_audio(self) -> bool:
        for index, item in enumerate(self._items):
            if isinstance(item, _AudioItem):
                del self._items[index]
                size = int(item.chunk.samples.size)
                self._audio_chunks -= 1
                self._samples -= size
                self.dropped_chunks += 1
                self.dropped_samples += size
                self._mark_gap()
                return True
        return False

    def _mark_gap(self) -> None:
        for index, item in enumerate(self._items):
            if isinstance(item, _AudioItem):
                self._items[index] = replace(item, forced_discontinuity=True)
                return
        self._pending_gap = True


class _Pipeline(Protocol):
    def start(self, *, timeout: float = 10.0) -> None: ...

    def stop(self, *, immediate: bool = False, timeout: float = 10.0) -> None: ...


class _SpeakerPipeline(_Pipeline, Protocol):
    def submit(self, data: np.ndarray) -> None: ...


class ActivatedWaveRecorder:
    """Persist finalized audio selected by the processor activation timeline."""

    def __init__(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if path.suffix.lower() != ".wav":
            raise ValueError("recording path must end with .wav")
        if not path.parent.is_dir():
            raise ValueError(f"recording directory does not exist: {path.parent}")
        self.path = path
        existing_samples = _wave_samples(path)
        self.samples = existing_samples or 0
        self.available = existing_samples is not None and existing_samples > 0
        self._writer: wave.Wave_write | None = None
        self._output: BinaryIO | None = None
        self._blocked_by_discontinuity = False

    @property
    def active(self) -> bool:
        return self._writer is not None

    def begin_session(self, history_samples: int) -> None:
        if history_samples <= 0:
            raise ValueError("recording history must be positive")
        self.finish_cycle()

    def accept_output(self, chunk: OutputChunk) -> None:
        if chunk.discontinuity:
            if self._writer is not None:
                self._finish(blocked_by_discontinuity=True)
        if self._blocked_by_discontinuity:
            return
        if self._writer is not None:
            self._write(chunk.samples)
            return
        if chunk.is_activated:
            self._start()
            self._write(chunk.samples)

    def finish_cycle(self) -> None:
        self._finish(blocked_by_discontinuity=False)

    def close(self) -> None:
        self._finish(blocked_by_discontinuity=self._blocked_by_discontinuity)

    def _start(self) -> None:
        output = self.path.open("w+b", buffering=0)
        try:
            writer = wave.open(output, "wb")
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
        except Exception:
            output.close()
            raise
        self._output = output
        self._writer = writer
        self.samples = 0
        self.available = False

    def _write(self, samples: np.ndarray) -> None:
        assert self._writer is not None
        self._writer.writeframes(samples.astype(np.dtype("<i2"), copy=False).tobytes())
        self.samples += int(samples.size)

    def _finish(self, *, blocked_by_discontinuity: bool) -> None:
        writer = self._writer
        output = self._output
        self._writer = None
        self._output = None
        if writer is not None:
            try:
                writer.close()
            finally:
                if output is not None:
                    output.close()
            self.available = self.samples > 0
        self._blocked_by_discontinuity = blocked_by_discontinuity


class DiagnosticCaptureHandler(CaptureHandler):
    def __init__(self, handoff: BoundedHandoff) -> None:
        self._handoff = handoff

    def on_chunk(self, chunk: CapturedChunk) -> None:
        self._handoff.put_audio(chunk)


class DiagnosticHistory:
    def __init__(self, history_samples: int, event_capacity: int) -> None:
        self._history_samples = history_samples
        self._event_capacity = event_capacity
        self.diagnostics: deque[DiagnosticEvent] = deque()
        self.events: deque[DisplayEvent] = deque(maxlen=event_capacity)

    def add_diagnostics(self, values: tuple[DiagnosticEvent, ...], current_sample: int) -> None:
        self.diagnostics.extend(values)
        for value in values:
            display = _display_event(value)
            if display is not None:
                self.events.append(display)
        cutoff = max(0, current_sample - self._history_samples)
        while self.diagnostics and _event_sample(self.diagnostics[0]) < cutoff:
            self.diagnostics.popleft()
        while len(self.diagnostics) > self._event_capacity:
            self.diagnostics.popleft()


class SessionController:
    """Own one real capture session and publish immutable snapshots."""

    def __init__(
        self,
        logger: Logger,
        *,
        recording_path: Path,
        processor_builder: Callable[[MicrophoneProfile, Logger, int], BuiltProcessor] | None = None,
        pipeline_factory: Callable[[CaptureHandler, str, Logger], _Pipeline] | None = None,
        speaker_factory: Callable[[str, Logger], _SpeakerPipeline] | None = None,
        handoff_chunks: int = 64,
        handoff_samples: int = 160_000,
        history_samples: int = DEFAULT_HISTORY_SAMPLES,
        event_capacity: int = DEFAULT_EVENT_CAPACITY,
    ) -> None:
        self._logger = logger.bind(module="wakelab", component="pipeline_lab")
        self._builder = processor_builder or _build
        self._pipeline_factory = pipeline_factory or _pipeline
        self._speaker_factory = speaker_factory or _speaker
        self._handoff_chunks = handoff_chunks
        self._handoff_samples = handoff_samples
        self._event_capacity = event_capacity
        self._history_samples = history_samples
        self._history = DiagnosticHistory(history_samples, event_capacity)
        self._recorder = ActivatedWaveRecorder(recording_path)
        self._lock = Lock()
        self._state = SessionState.READY
        self._owner_id: str | None = None
        self._profile_name: str | None = None
        self._generation: int | None = None
        self._current_sample = 0
        self._retained_samples = 0
        self._diagnostic_drops = 0
        self._latency_samples: deque[tuple[int, float, float]] = deque()
        self._capture_latency_ms: float | None = None
        self._latency_ms: float | None = None
        self._vad_probability: float | None = None
        self._vad_is_speech = False
        self._wake_score: float | None = None
        self._wake_hit_count = 0
        self._wake_state = "armed"
        self._activated = False
        self._recording_samples = self._recorder.samples
        self._recording_active = False
        self._recording_available = self._recorder.available
        self._playback_active = False
        self._providers: tuple[str, ...] = ()
        self._error: str | None = None
        self._handoff: BoundedHandoff | None = None
        self._processor: WakeStreamProcessor | None = None
        self._pipeline: _Pipeline | None = None
        self._worker: Thread | None = None
        self._cancelled = Event()
        self._playback_cancelled = Event()
        self._playback_reserved = False
        self._speaker: _SpeakerPipeline | None = None

    def start(self, owner_id: str, device_id: str, profile: MicrophoneProfile) -> None:
        with self._lock:
            if self._state in {SessionState.LOADING, SessionState.LISTENING, SessionState.STOPPING} or (
                self._worker is not None and self._worker.is_alive()
            ):
                raise RuntimeError("a diagnostic session is already active")
            self._recorder.begin_session(profile.stream_config.max_retained_audio_samples)
            self._state = SessionState.LOADING
            self._owner_id = owner_id
            self._profile_name = profile.name
            self._error = None
            self._cancelled.clear()
            self._history = DiagnosticHistory(self._history_samples, self._event_capacity)
            self._generation = None
            self._current_sample = 0
            self._retained_samples = 0
            self._diagnostic_drops = 0
            self._latency_samples.clear()
            self._capture_latency_ms = None
            self._latency_ms = None
            self._vad_probability = None
            self._vad_is_speech = False
            self._wake_score = None
            self._wake_hit_count = 0
            self._wake_state = "armed"
            self._activated = False
            self._recording_samples = self._recorder.samples
            self._recording_active = self._recorder.active
            self._recording_available = self._recorder.available
            self._providers = ()
            self._handoff = None
            self._processor = None
            self._pipeline = None
            self._worker = None
        built: BuiltProcessor | None = None
        try:
            built = self._builder(profile, self._logger, self._event_capacity)
            if self._cancelled.is_set():
                built.processor.close()
                with self._lock:
                    if self._state is not SessionState.ERROR:
                        self._state = SessionState.STOPPED
                return
            handoff = BoundedHandoff(self._handoff_chunks, self._handoff_samples)
            handler = DiagnosticCaptureHandler(handoff)
            pipeline = self._pipeline_factory(handler, device_id, self._logger)
            worker = Thread(target=self._run_dsp, args=(handoff, built.processor), name="wake-pipeline-dsp")
            with self._lock:
                self._processor = built.processor
                self._handoff = handoff
                self._pipeline = pipeline
                self._worker = worker
                self._providers = built.providers
            worker.start()
            if self._cancelled.is_set():
                handoff.put_command("stop")
                pipeline.stop(immediate=True)
                worker.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
                if worker.is_alive():
                    with self._lock:
                        self._state = SessionState.STOPPING
                return
            pipeline.start()
            with self._lock:
                if not self._cancelled.is_set():
                    self._state = SessionState.LISTENING
        except Exception as error:
            self._fail(error)
            with self._lock:
                published = built is not None and self._processor is built.processor
            if built is not None and not published:
                try:
                    built.processor.close()
                except Exception:
                    pass
            self.stop(immediate=True)
            raise

    def rearm(self) -> None:
        with self._lock:
            handoff = self._handoff if self._state is SessionState.LISTENING else None
        if handoff is None:
            raise RuntimeError("no active processor to re-arm")
        command = handoff.put_command("rearm")
        if command is None or not command.completed.wait(timeout=10):
            raise RuntimeError("processor did not complete re-arm")
        if command.error is not None:
            raise RuntimeError("processor failed to re-arm") from command.error

    def play(self, device_id: str) -> bool:
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("speaker device ID must not be empty")
        with self._lock:
            if self._playback_reserved:
                raise RuntimeError("recording playback is already active")
            if self._recording_active or not self._recording_available:
                raise RuntimeError("no finalized activation recording is available")
            samples = _read_wave(self._recorder.path)
            self._playback_cancelled.clear()
            self._playback_reserved = True
            self._playback_active = True

        speaker: _SpeakerPipeline | None = None
        failure: Exception | None = None
        try:
            speaker = self._speaker_factory(device_id, self._logger)
            with self._lock:
                self._speaker = speaker
            if self._playback_cancelled.is_set():
                return False
            speaker.start()
            if self._playback_cancelled.is_set():
                return False
            speaker.submit(samples)
        except PlaybackSubmissionError as error:
            if not self._playback_cancelled.is_set():
                failure = error
        except Exception as error:
            failure = error
        finally:
            cancelled = self._playback_cancelled.is_set()
            if speaker is not None:
                try:
                    speaker.stop(immediate=cancelled or failure is not None)
                except Exception as error:
                    if failure is None:
                        failure = error
            with self._lock:
                if self._speaker is speaker:
                    self._speaker = None
                self._playback_reserved = False
                self._playback_active = False
        if failure is not None and not self._playback_cancelled.is_set():
            raise failure
        return not self._playback_cancelled.is_set()

    def stop_playback(self) -> None:
        self._playback_cancelled.set()
        with self._lock:
            speaker = self._speaker
        if speaker is not None:
            with suppress(Exception):
                speaker.stop(immediate=True)

    def request_stop(self) -> None:
        self._cancelled.set()
        with self._lock:
            if self._state in {SessionState.LOADING, SessionState.LISTENING}:
                self._state = SessionState.STOPPING
            handoff = self._handoff
        if handoff is not None:
            handoff.put_command("stop")

    def stop(self, *, immediate: bool = True) -> None:
        self.request_stop()
        with self._lock:
            pipeline = self._pipeline
            worker = self._worker
        if pipeline is not None:
            try:
                pipeline.stop(immediate=immediate)
            except Exception as error:
                self._fail(error)
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        with self._lock:
            if worker is not None and worker.is_alive():
                if self._state is not SessionState.ERROR:
                    self._state = SessionState.STOPPING
                self._error = self._error or "diagnostic worker did not stop before the timeout"
            elif self._state is not SessionState.ERROR:
                self._state = SessionState.STOPPED

    def shutdown(self) -> None:
        self.stop_playback()
        self.stop(immediate=True)

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            handoff = self._handoff
            return SessionSnapshot(
                self._state,
                self._owner_id,
                self._profile_name,
                self._generation,
                self._current_sample,
                self._retained_samples,
                handoff.depth if handoff is not None else 0,
                handoff.samples if handoff is not None else 0,
                handoff.dropped_chunks if handoff is not None else 0,
                handoff.dropped_samples if handoff is not None else 0,
                self._diagnostic_drops,
                self._capture_latency_ms,
                self._latency_ms,
                self._vad_probability,
                self._vad_is_speech,
                self._wake_score,
                self._wake_hit_count,
                self._wake_state,
                self._activated,
                str(self._recorder.path),
                self._recording_samples,
                self._recording_active,
                self._recording_available,
                self._playback_active,
                self._providers,
                tuple(self._history.diagnostics),
                tuple(self._history.events),
                self._error,
            )

    def _run_dsp(self, handoff: BoundedHandoff, processor: WakeStreamProcessor) -> None:
        previous_generation: int | None = None
        try:
            while (item := handoff.get()) is not None:
                if isinstance(item, _Command):
                    if item.name == "stop":
                        break
                    if item.name == "rearm":
                        try:
                            self._record_outputs(processor.rearm())
                            self._collect(processor)
                            self._finish_recording_cycle()
                        except BaseException as error:
                            item.error = error
                            raise
                        finally:
                            item.completed.set()
                    continue
                chunk = item.chunk
                discontinuity = (
                    chunk.discontinuity
                    or item.forced_discontinuity
                    or (previous_generation is not None and chunk.generation != previous_generation)
                )
                before = time.perf_counter_ns()
                outputs = processor.process(
                    InputChunk(
                        chunk.samples,
                        chunk.running_time_ns,
                        chunk.captured_at_ns,
                        chunk.generation,
                        discontinuity,
                    )
                )
                self._record_outputs(outputs)
                previous_generation = chunk.generation
                after = time.perf_counter_ns()
                latency = (after - before) / 1_000_000
                with self._lock:
                    self._generation = chunk.generation
                    self._current_sample = processor.accepted_samples
                    self._retained_samples = processor.retained_samples
                    self._latency_samples.append((after, item.capture_latency_ms, latency))
                    cutoff = after - LATENCY_AVERAGE_WINDOW_NS
                    while self._latency_samples[0][0] < cutoff:
                        self._latency_samples.popleft()
                    count = len(self._latency_samples)
                    self._capture_latency_ms = sum(value[1] for value in self._latency_samples) / count
                    self._latency_ms = sum(value[2] for value in self._latency_samples) / count
                self._collect(processor)
        except Exception as error:
            self._fail(error)
            with self._lock:
                pipeline = self._pipeline
            if pipeline is not None:
                try:
                    pipeline.stop(immediate=True)
                except Exception:
                    pass
        finally:
            try:
                self._record_outputs(processor.close())
            except StreamCloseError as error:
                self._record_outputs(error.outputs)
                self._fail(error)
            except Exception as error:
                self._fail(error)
            self._collect(processor)
            try:
                self._finish_recording_cycle()
            except Exception as error:
                self._fail(error)
            handoff.close()
            with self._lock:
                if self._worker is current_thread():
                    self._worker = None
                    if self._state is SessionState.STOPPING:
                        self._state = SessionState.STOPPED

    def _sync_recording(self) -> None:
        with self._lock:
            self._recording_samples = self._recorder.samples
            self._recording_active = self._recorder.active
            self._recording_available = self._recorder.available

    def _finish_recording_cycle(self) -> None:
        self._recorder.finish_cycle()
        with self._lock:
            self._recording_samples = self._recorder.samples
            self._recording_active = False
            self._recording_available = self._recorder.available

    def _record_outputs(self, outputs: list[OutputChunk]) -> None:
        for output in outputs:
            self._recorder.accept_output(output)
        self._sync_recording()

    def _collect(self, processor: WakeStreamProcessor) -> None:
        drained = processor.drain_diagnostics()
        with self._lock:
            self._diagnostic_drops += drained.dropped_events
            for event in drained.events:
                self._update_current(event)
            self._history.add_diagnostics(drained.events, processor.accepted_samples)

    def _update_current(self, event: DiagnosticEvent) -> None:
        if isinstance(event, VadFrameDiagnostic):
            self._vad_probability = event.probability
        elif isinstance(event, VadSpeechDiagnostic):
            self._vad_is_speech = event.is_speech
        elif isinstance(event, WakeCandidateStartedDiagnostic):
            self._wake_state = "evaluating"
            self._wake_score = None
            self._wake_hit_count = 0
        elif isinstance(event, WakeFrameDiagnostic):
            self._wake_score = event.score
            self._wake_hit_count = event.consecutive_hit_count
        elif isinstance(event, WakeCandidateEndedDiagnostic):
            self._wake_state = "armed"
            self._wake_score = None
            self._wake_hit_count = 0
        elif isinstance(event, ActivationDiagnostic):
            self._activated = True
            self._wake_state = "bypassed"
            self._wake_score = None
            self._wake_hit_count = 0
        elif isinstance(event, RearmDiagnostic):
            self._activated = False
            self._wake_state = "armed"
            self._wake_score = None
            self._wake_hit_count = 0
        elif isinstance(event, ContinuityBoundaryDiagnostic):
            self._vad_probability = None
            self._vad_is_speech = False
            self._wake_state = "bypassed" if self._activated else "armed"
            self._wake_score = None
            self._wake_hit_count = 0

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            self._state = SessionState.ERROR
            self._error = str(error)
        self._logger.error("pipeline_lab_session_failed", error=str(error))


def _build(profile: MicrophoneProfile, logger: Logger, capacity: int) -> BuiltProcessor:
    return build_processor(profile, logger, diagnostic_event_capacity=capacity)


def _pipeline(handler: CaptureHandler, device_id: str, logger: Logger) -> _Pipeline:
    return MicrophoneCapturePipeline(
        logger=logger,
        handler=handler,
        audio_format=AudioFormat(sample_rate=SAMPLE_RATE, channels=1),
        device_id=device_id,
    )


def _speaker(device_id: str, logger: Logger) -> _SpeakerPipeline:
    return SpeakerPlaybackPipeline(
        logger=logger,
        audio_format=AudioFormat(sample_rate=SAMPLE_RATE, channels=1),
        device_id=device_id,
    )


def _wave_samples(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with wave.open(str(path), "rb") as recording:
            if (
                recording.getnchannels() != 1
                or recording.getsampwidth() != 2
                or recording.getframerate() != SAMPLE_RATE
            ):
                return None
            return recording.getnframes()
    except (OSError, EOFError, wave.Error):
        return None


def _read_wave(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as recording:
        if recording.getnchannels() != 1 or recording.getsampwidth() != 2 or recording.getframerate() != SAMPLE_RATE:
            raise ValueError("activation recording must be 16 kHz mono PCM16 WAV")
        samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype=np.dtype("<i2")).copy()
    if samples.size == 0:
        raise ValueError("activation recording is empty")
    return samples


def _event_sample(event: DiagnosticEvent) -> int:
    if isinstance(event, VadFrameDiagnostic | WakeFrameDiagnostic):
        return event.end_sample
    if isinstance(event, WakeCandidateStartedDiagnostic):
        return event.start_sample
    if isinstance(event, WakeCandidateEndedDiagnostic):
        return event.end_sample
    if isinstance(event, ActivationDiagnostic):
        return event.confirming_end_sample
    if isinstance(event, RearmDiagnostic):
        return event.effective_sample
    if isinstance(event, ContinuityBoundaryDiagnostic):
        return event.sample
    if isinstance(event, FailureDiagnostic):
        return event.retained_end_sample
    return event.sample


def _display_event(event: DiagnosticEvent) -> DisplayEvent | None:
    if isinstance(event, VadFrameDiagnostic):
        return None
    sample = _event_sample(event)
    if isinstance(event, WakeFrameDiagnostic):
        if event.consecutive_hit_count == 0:
            return None
        return DisplayEvent(sample, "wake_hit", f"score {event.score:.4f}, streak {event.consecutive_hit_count}")
    if isinstance(event, VadSpeechDiagnostic):
        return DisplayEvent(sample, "speech_start" if event.is_speech else "speech_end", event.reason.value)
    if isinstance(event, WakeCandidateStartedDiagnostic):
        return DisplayEvent(sample, "candidate_start", "wake evaluation started")
    if isinstance(event, WakeCandidateEndedDiagnostic):
        return DisplayEvent(sample, "candidate_end", event.reason.value)
    if isinstance(event, ActivationDiagnostic):
        return DisplayEvent(sample, "activation", f"score {event.confirming_score:.4f}")
    if isinstance(event, RearmDiagnostic):
        return DisplayEvent(sample, "rearm", "activation cleared")
    if isinstance(event, ContinuityBoundaryDiagnostic):
        return DisplayEvent(sample, "boundary", event.reason.value)
    return DisplayEvent(sample, "error", event.message)
