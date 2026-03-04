"""Synchronous assembly of framing, policies, and public output chunks."""

from __future__ import annotations

from copy import deepcopy
from collections import deque

from lumivox_core.logger import Logger

from .stream import InputChunk, OutputChunk, StreamConfig, StreamCloseError, StreamStateError, StreamProcessingError
from ._framing import VadFramer, WakeWordFramer
from ._timeline import SampleRange, AudioTimeline
from ._activation import ActivationRange, ActivationLifecycle
from .vad._policy import VadRange, VadPolicy
from .vad._backend import VadBackend
from .wakeword._policy import WakePolicy, WakeDecision
from .wakeword._backend import WakeWordBackend


class WakeStreamProcessor:
    """Process one PCM stream with injected VAD and wake-word backends."""

    def __init__(
        self, config: StreamConfig, vad_backend: VadBackend, wake_backend: WakeWordBackend, /, *, logger: Logger
    ) -> None:
        if not isinstance(config, StreamConfig):
            raise TypeError("config must be a StreamConfig")
        self._vad_framer = VadFramer(vad_backend)
        self._wake_framer = WakeWordFramer(wake_backend)
        required = max(
            self._vad_framer.frame_samples + config.vad_policy.minimum_speech_samples,
            self._vad_framer.frame_samples
            + config.vad_policy.minimum_silence_samples
            + config.vad_policy.right_padding_samples,
            self._vad_framer.frame_samples + self._wake_framer.frame_samples + config.wake_policy.pre_roll_samples,
        )
        if config.max_retained_audio_samples < required:
            raise ValueError(f"max_retained_audio_samples must be at least {required}")

        self._timeline = AudioTimeline(config.max_retained_audio_samples)
        self._vad_policy = VadPolicy(config.vad_policy)
        self._wake_policy = WakePolicy(config.wake_policy, self._wake_framer)
        self._activation = ActivationLifecycle()
        self._pending_vad: deque[VadRange] = deque()
        self._pending_decision: WakeDecision | None = None
        self._generation: int | None = None
        self._segment_active = False
        self._finished = False
        self._closed = False
        self._failed = False
        self._failure_outputs: list[OutputChunk] = []
        self._failure_tail: InputChunk | None = None
        self._accepted_samples = 0
        self._emitted_samples = 0
        self._logger = logger.bind(module="wakelab", component="stream_processor")
        self._logger.info("stream_processor_initialized", retained_audio_capacity=config.max_retained_audio_samples)

    @property
    def accepted_samples(self) -> int:
        """Return the processor-lifetime count of accepted samples."""

        return self._accepted_samples

    @property
    def emitted_samples(self) -> int:
        """Return the processor-lifetime count of emitted samples."""

        return self._emitted_samples

    @property
    def retained_samples(self) -> int:
        """Return the amount of currently undecided PCM."""

        return self._timeline.retained_samples

    def process(self, chunk: InputChunk, /) -> list[OutputChunk]:
        """Accept a valid chunk and return every range with final metadata."""

        if self._closed or self._finished or self._failed:
            raise StreamStateError("processor is no longer accepting input")
        if not isinstance(chunk, InputChunk):
            raise TypeError("chunk must be an InputChunk")

        # Ownership happens before any reset or inference, so a later failure can
        # conservatively release the whole caller chunk rather than a prefix.
        owned = InputChunk(
            chunk.samples.copy(),
            chunk.running_time_ns,
            chunk.captured_at_ns,
            chunk.generation,
            chunk.discontinuity,
        )
        self._accepted_samples += owned.samples.size
        outputs: list[OutputChunk] = []
        offset = 0
        try:
            if self._segment_active and (owned.discontinuity or owned.generation != self._generation):
                self._logger.info("stream_boundary", generation=owned.generation, position=self._timeline.next_position)
                outputs.extend(self._finish_segment())
            if not self._segment_active:
                self._start_segment(self._timeline.next_position, owned.generation)

            while offset < owned.samples.size:
                free = self._timeline.capacity - self._timeline.retained_samples
                if free == 0:
                    outputs.extend(self._advance())
                    free = self._timeline.capacity - self._timeline.retained_samples
                    if free == 0:
                        raise BufferError("configured retained-audio bound cannot advance the stream")
                length = min(free, owned.samples.size - offset)
                part = self._part(owned, offset, length)
                self._timeline.append(part)
                offset += length
                outputs.extend(self._advance())
        except Exception as exc:
            self._failed = True
            self._failure_outputs = outputs
            if offset < owned.samples.size:
                self._failure_tail = self._part(owned, offset, owned.samples.size - offset)
            self._logger.error(
                "stream_processing_failed",
                generation=owned.generation,
                position=self._timeline.next_position,
                pending_samples=self._timeline.retained_samples,
                error=str(exc),
            )
            raise StreamProcessingError("processing failed after accepting the complete input chunk") from exc
        return self._release(outputs)

    def rearm(self) -> list[OutputChunk]:
        """Cancel wake evaluation and clear the activation latch for future audio."""

        self._require_open()
        if not self._segment_active:
            return []
        outputs = self._advance()
        # Pending VAD evidence is not yet public, so it belongs to the new armed state.
        if self._wake_policy.evaluating:
            self._logger.info("wake_candidate_rejected", position=self._activation_position(), reason="rearmed")
        self._wake_policy.rearm()
        self._activation.rearm(self._activation_position())
        self._logger.info("stream_rearmed", position=self._activation_position(), pending_samples=self.retained_samples)
        return self._release(outputs)

    def discontinue(self) -> list[OutputChunk]:
        """Finalize the current continuity segment without ending the processor."""

        self._require_open()
        outputs = self._finish_segment() if self._segment_active else []
        self._logger.info("stream_discontinued", position=self._timeline.next_position)
        return self._release(outputs)

    def finish(self) -> list[OutputChunk]:
        """Finalize all real audio and permanently reject further input."""

        self._require_open()
        outputs = self._finish_segment() if self._segment_active else []
        self._finished = True
        self._logger.info("stream_finished", position=self._timeline.next_position)
        return self._release(outputs)

    def drain_failed(self) -> list[OutputChunk]:
        """Conservatively release accepted audio after a terminal processing failure."""

        if not self._failed:
            raise StreamStateError("processor has not entered a terminal failure state")
        outputs = self._failure_outputs
        self._failure_outputs = []
        tail_is_speech = False
        if self._segment_active:
            end = self._timeline.next_position
            self._pending_vad.extend(self._vad_policy.finish(end))
            tail_is_speech = self._vad_policy.failure_is_speech
            if self._wake_policy.evaluating:
                self._logger.info("wake_candidate_rejected", position=end, reason="processing_failed")
            outputs.extend(self._emit_through(end, evaluating=False))
            self._activation.finish_segment(end)
            self._segment_active = False
            self._generation = None
        if self._failure_tail is not None:
            tail = self._failure_tail
            self._failure_tail = None
            outputs.append(
                OutputChunk(
                    tail.samples.copy(),
                    tail.running_time_ns,
                    tail.captured_at_ns,
                    tail.generation,
                    tail.discontinuity,
                    tail_is_speech,
                    self._activation.state.value == "activated",
                )
            )
        self._logger.info("stream_failure_drained", accepted_samples=self.accepted_samples, pending_samples=0)
        return self._release(outputs)

    def close(self) -> list[OutputChunk]:
        """Finalize retained audio, close owned backends, and become terminal."""

        if self._closed:
            return []
        try:
            outputs = self.drain_failed() if self._failed else (self.finish() if not self._finished else [])
        except Exception as exc:
            self._failed = True
            self._logger.error(
                "stream_finalization_failed",
                position=self._timeline.next_position,
                pending_samples=self._timeline.retained_samples,
                error=str(exc),
            )
            outputs = self.drain_failed()
        errors: list[BaseException] = []
        for component, framer in (("vad", self._vad_framer), ("wake", self._wake_framer)):
            try:
                framer.close()
            except Exception as exc:
                errors.append(exc)
                self._logger.error("stream_resource_close_failed", component=component, error=str(exc))
        self._closed = True
        self._logger.info("stream_closed", pending_samples=self.retained_samples)
        if errors:
            raise StreamCloseError("resource closure failed", outputs, tuple(errors))
        return outputs

    def _start_segment(self, start: int, generation: int) -> None:
        self._vad_framer.start_segment(start)
        self._vad_policy.start_segment(start)
        self._wake_policy.start_segment(start)
        self._activation.start_segment(start)
        self._generation = generation
        self._segment_active = True

    def _finish_segment(self) -> list[OutputChunk]:
        end = self._timeline.next_position
        ranges = self._vad_policy.finish(end)
        self._consume_vad(ranges)
        if self._wake_policy.evaluating:
            self._logger.info("wake_candidate_rejected", position=end)
        self._wake_policy.finish(end)
        outputs = self._emit_through(end)
        self._activation.finish_segment(end)
        self._segment_active = False
        self._generation = None
        return outputs

    def _advance(self) -> list[OutputChunk]:
        results = self._vad_framer.consume(self._timeline, self._timeline.next_position)
        self._consume_vad(self._vad_policy.consume(results))
        return self._emit_through(self._wake_policy.safe_frontier or self._timeline.retained_range.start)

    def _consume_vad(self, ranges: list[VadRange]) -> None:
        for vad_range in ranges:
            self._pending_vad.append(vad_range)
            update = self._wake_policy.consume(self._timeline, [vad_range])
            if update.decision is not None:
                self._pending_decision = update.decision
                self._wake_policy.activate()
                self._logger.info(
                    "wake_activated",
                    position=update.decision.activation_start,
                    generation=self._generation,
                )

    def _emit_through(self, frontier: int, *, evaluating: bool | None = None) -> list[OutputChunk]:
        pending_before = self._pending_vad.copy()
        decision_before = self._pending_decision
        activation_before = deepcopy(self._activation)
        ranges: list[VadRange] = []
        while self._pending_vad and self._pending_vad[0].sample_range.start < frontier:
            item = self._pending_vad.popleft()
            end = min(item.sample_range.end, frontier)
            ranges.append(VadRange(SampleRange(item.sample_range.start, end), item.is_speech))
            if end < item.sample_range.end:
                self._pending_vad.appendleft(VadRange(SampleRange(end, item.sample_range.end), item.is_speech))
        if not ranges:
            return []
        decision = self._pending_decision
        if decision is not None and decision.activation_start <= ranges[-1].sample_range.end:
            self._pending_decision = None
        else:
            decision = None
        try:
            activated = self._activation.consume(
                ranges, decision, evaluating=self._wake_policy.evaluating if evaluating is None else evaluating
            )
            outputs = [self._output(item) for item in activated]
        except Exception:
            self._pending_vad = pending_before
            self._pending_decision = decision_before
            self._activation = activation_before
            raise
        self._timeline.release_before(ranges[-1].sample_range.end)
        return outputs

    def _output(self, item: ActivationRange) -> OutputChunk:
        start = item.sample_range.start
        metadata = self._timeline.metadata_at(start)
        return OutputChunk(
            self._timeline.read(start, item.sample_range.end),
            metadata.running_time_ns,
            metadata.captured_at_ns,
            metadata.generation,
            metadata.discontinuity,
            item.is_speech,
            item.is_activated,
        )

    def _activation_position(self) -> int:
        if self._pending_vad:
            return self._pending_vad[0].sample_range.start
        return self._timeline.next_position

    def _require_open(self) -> None:
        if self._closed or self._finished or self._failed:
            raise StreamStateError("processor is closed or finished")

    def _release(self, outputs: list[OutputChunk]) -> list[OutputChunk]:
        self._emitted_samples += sum(output.samples.size for output in outputs)
        return outputs

    @staticmethod
    def _part(chunk: InputChunk, offset: int, length: int) -> InputChunk:
        running_time = 0 if chunk.running_time_ns == 0 else chunk.running_time_ns + offset * 1_000_000_000 // 16_000
        return InputChunk(
            chunk.samples[offset : offset + length],
            running_time,
            chunk.captured_at_ns + offset * 1_000_000_000 // 16_000,
            chunk.generation,
            chunk.discontinuity and offset == 0,
        )
