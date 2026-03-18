"""Synchronous assembly of framing, policies, and public output chunks."""

from __future__ import annotations

from copy import deepcopy
from contextlib import suppress
from collections import deque

from lumivox_core.logger import Logger

from .stream import InputChunk, OutputChunk, StreamConfig, StreamCloseError, StreamStateError, StreamProcessingError
from ._framing import VadFramer, WakeWordFramer
from ._timeline import SampleRange, AudioTimeline
from ._retention import required_retained_audio_samples
from ._activation import ActivationRange, ActivationLifecycle
from .diagnostics import (
    DiagnosticDrain,
    RearmDiagnostic,
    FailureDiagnostic,
    VadFrameDiagnostic,
    VadSpeechDiagnostic,
    VadTransitionReason,
    ActivationDiagnostic,
    WakeCandidateEndReason,
    ContinuityBoundaryReason,
    DiagnosticFailureOperation,
    ContinuityBoundaryDiagnostic,
)
from .vad._policy import VadRange, VadPolicy
from ._diagnostics import DiagnosticCollector
from .vad._backend import VadBackend
from .wakeword._policy import WakePolicy, WakeDecision
from .wakeword._backend import WakeWordBackend


class WakeStreamProcessor:
    """Process one PCM stream with injected VAD and wake-word backends."""

    def __init__(
        self,
        config: StreamConfig,
        vad_backend: VadBackend,
        wake_backend: WakeWordBackend,
        /,
        *,
        logger: Logger,
        diagnostic_event_capacity: int = 0,
    ) -> None:
        if not isinstance(config, StreamConfig):
            raise TypeError("config must be a StreamConfig")
        if isinstance(diagnostic_event_capacity, bool) or not isinstance(diagnostic_event_capacity, int):
            raise TypeError("diagnostic_event_capacity must be an integer")
        if diagnostic_event_capacity < 0:
            raise ValueError("diagnostic_event_capacity must be nonnegative")
        self._diagnostics = DiagnosticCollector(diagnostic_event_capacity) if diagnostic_event_capacity else None
        self._vad_framer = VadFramer(vad_backend)
        self._wake_framer = WakeWordFramer(wake_backend)
        required = required_retained_audio_samples(
            config,
            self._vad_framer.frame_samples,
            self._wake_framer.frame_samples,
        )
        if config.max_retained_audio_samples < required:
            raise ValueError(f"max_retained_audio_samples must be at least {required}")

        self._timeline = AudioTimeline(config.max_retained_audio_samples)
        self._vad_policy = VadPolicy(config.vad_policy)
        self._wake_policy = WakePolicy(config.wake_policy, self._wake_framer, diagnostics=self._diagnostics)
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
        self._failure_tail_is_speech = False
        self._failure_drained = False
        self._final_vad_ranges: list[VadRange] | None = None
        self._vad_finalization_attempted = False
        self._final_vad_consumed = 0
        self._wake_finalized = False
        self._segment_output_assembled = False
        self._activation_finalized = False
        self._accepted_samples = 0
        self._emitted_samples = 0
        self._diagnostic_vad_speech = False
        self._force_next_discontinuity = False
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

    def drain_diagnostics(self) -> DiagnosticDrain:
        """Return and clear bounded diagnostic events without affecting audio."""

        if self._diagnostics is None:
            return DiagnosticDrain((), 0)
        return self._diagnostics.drain()

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
        forced_discontinuity = self._force_next_discontinuity
        if forced_discontinuity:
            owned = InputChunk(
                owned.samples,
                owned.running_time_ns,
                owned.captured_at_ns,
                owned.generation,
                True,
            )
            self._force_next_discontinuity = False
        self._accepted_samples += owned.samples.size
        outputs: list[OutputChunk] = []
        offset = 0
        try:
            boundary = owned.discontinuity or (self._segment_active and owned.generation != self._generation)
            if self._segment_active and boundary:
                self._logger.info("stream_boundary", generation=owned.generation, position=self._timeline.next_position)
                previous_generation = self._generation
                outputs.extend(self._finish_segment(WakeCandidateEndReason.CONTINUITY_BOUNDARY))
                self._record_boundary(previous_generation, owned.generation, owned.discontinuity)
            elif not self._segment_active and owned.discontinuity and not forced_discontinuity:
                self._record_boundary(None, owned.generation, True)
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
            tail = self._part(owned, offset, owned.samples.size - offset) if offset < owned.samples.size else None
            self._enter_failure(exc, outputs, DiagnosticFailureOperation.PROCESS, tail=tail)
            raise StreamProcessingError("processing failed after accepting the complete input chunk") from exc
        return self._release(outputs)

    def rearm(self) -> list[OutputChunk]:
        """Cancel wake evaluation and clear the activation latch for future audio."""

        self._require_open()
        outputs: list[OutputChunk] = []
        try:
            outputs.extend(self._advance() if self._segment_active else [])
            boundary = self._timeline.next_position
            # Pending VAD evidence is not yet public, so it belongs to the new armed state.
            if self._wake_policy.evaluating:
                self._logger.info("wake_candidate_rejected", position=boundary, reason="rearmed")
            self._wake_policy.rearm(boundary)
            self._activation.rearm(boundary)
            if self._segment_active:
                outputs.extend(self._emit_through(boundary, evaluating=False))
            if self._diagnostics is not None:
                self._diagnostics.append(RearmDiagnostic(self._generation, boundary))
            self._logger.info("stream_rearmed", position=boundary, pending_samples=self.retained_samples)
        except Exception as exc:
            self._enter_failure(exc, outputs, DiagnosticFailureOperation.FINALIZE)
            raise StreamProcessingError("re-arm failed and made the processor terminal") from exc
        return self._release(outputs)

    def discontinue(self) -> list[OutputChunk]:
        """Finalize the current continuity segment without ending the processor."""

        self._require_open()
        outputs: list[OutputChunk] = []
        try:
            previous_generation = self._generation
            if self._segment_active:
                outputs.extend(self._finish_segment(WakeCandidateEndReason.CONTINUITY_BOUNDARY))
            self._force_next_discontinuity = True
            if self._diagnostics is not None:
                self._diagnostics.append(
                    ContinuityBoundaryDiagnostic(
                        self._timeline.next_position,
                        previous_generation,
                        None,
                        ContinuityBoundaryReason.EXPLICIT_DISCONTINUE,
                    )
                )
            self._logger.info("stream_discontinued", position=self._timeline.next_position)
        except Exception as exc:
            self._enter_failure(exc, outputs, DiagnosticFailureOperation.FINALIZE)
            raise StreamProcessingError("discontinuity finalization failed and made the processor terminal") from exc
        return self._release(outputs)

    def finish(self) -> list[OutputChunk]:
        """Finalize all real audio and permanently reject further input."""

        if self._finished:
            return []
        self._require_open()
        outputs: list[OutputChunk] = []
        try:
            if self._segment_active:
                outputs.extend(self._finish_segment(WakeCandidateEndReason.FINISHED))
            self._finished = True
            self._logger.info("stream_finished", position=self._timeline.next_position)
        except Exception as exc:
            self._enter_failure(exc, outputs, DiagnosticFailureOperation.FINALIZE)
            raise StreamProcessingError("stream finalization failed and made the processor terminal") from exc
        return self._release(outputs)

    def drain_failed(self) -> list[OutputChunk]:
        """Conservatively release accepted audio after a terminal processing failure."""

        if not self._failed:
            raise StreamStateError("processor has not entered a terminal failure state")
        if self._failure_drained:
            return []
        if self._segment_active:
            end = self._timeline.next_position
            if self._final_vad_ranges is None:
                self._failure_tail_is_speech = self._vad_policy.failure_is_speech
                if not self._vad_finalization_attempted:
                    self._vad_finalization_attempted = True
                    try:
                        self._final_vad_ranges = self._vad_policy.finish(end)
                    except Exception:
                        pass
                    else:
                        self._record_vad_ranges(self._final_vad_ranges, VadTransitionReason.PROCESSING_FAILURE)
                if self._final_vad_ranges is None:
                    self._final_vad_ranges = []
                    start = (
                        self._pending_vad[-1].sample_range.end
                        if self._pending_vad
                        else self._timeline.retained_range.start
                    )
                    if start < end:
                        self._pending_vad.append(VadRange(SampleRange(start, end), self._failure_tail_is_speech))
            while self._final_vad_consumed < len(self._final_vad_ranges):
                self._pending_vad.append(self._final_vad_ranges[self._final_vad_consumed])
                self._final_vad_consumed += 1
            if not self._wake_finalized:
                if self._wake_policy.evaluating:
                    self._logger.info("wake_candidate_rejected", position=end, reason="processing_failed")
                try:
                    self._wake_policy.finish(end, reason=WakeCandidateEndReason.PROCESSING_FAILED)
                finally:
                    self._wake_finalized = True
            if not self._segment_output_assembled:
                assembled = self._emit_through(end, evaluating=False)
                self._failure_outputs.extend(assembled)
                self._segment_output_assembled = True
            if not self._activation_finalized:
                try:
                    self._activation.finish_segment(end)
                finally:
                    self._activation_finalized = True
            self._segment_active = False
            self._generation = None
        if self._failure_tail is not None:
            tail = self._failure_tail
            output = OutputChunk(
                tail.samples.copy(),
                tail.running_time_ns,
                tail.captured_at_ns,
                tail.generation,
                tail.discontinuity,
                self._failure_tail_is_speech,
                self._activation.wake_inference_bypassed,
            )
            self._failure_outputs.append(output)
            self._failure_tail = None
        outputs = self._failure_outputs
        with suppress(Exception):
            self._logger.info("stream_failure_drained", accepted_samples=self.accepted_samples, pending_samples=0)
        self._failure_outputs = []
        self._failure_drained = True
        return self._release(outputs)

    def close(self) -> list[OutputChunk]:
        """Finalize retained audio, close owned backends, and become terminal."""

        if self._closed:
            return []
        errors: list[BaseException] = []
        outputs: list[OutputChunk] = []
        try:
            outputs = self.drain_failed() if self._failed else (self.finish() if not self._finished else [])
        except Exception as exc:
            errors.append(exc)
            if not self._failed:
                self._enter_failure(exc, [], DiagnosticFailureOperation.FINALIZE)
            try:
                outputs.extend(self.drain_failed())
            except Exception as drain_error:
                errors.append(drain_error)
        for component, framer in (("vad", self._vad_framer), ("wake", self._wake_framer)):
            try:
                framer.close()
            except Exception as exc:
                errors.append(exc)
                with suppress(Exception):
                    self._logger.error("stream_resource_close_failed", component=component, error=str(exc))
                    if self._diagnostics is not None:
                        operation = (
                            DiagnosticFailureOperation.VAD_CLOSE
                            if component == "vad"
                            else DiagnosticFailureOperation.WAKE_CLOSE
                        )
                        self._diagnostics.append(
                            FailureDiagnostic(
                                self._generation,
                                self._timeline.next_position,
                                self._accepted_samples,
                                operation,
                                type(exc).__name__,
                                str(exc),
                            )
                        )
        self._closed = True
        with suppress(Exception):
            self._logger.info("stream_closed", pending_samples=self.retained_samples)
        if errors:
            raise StreamCloseError("resource closure failed", outputs, tuple(errors))
        return outputs

    def _start_segment(self, start: int, generation: int) -> None:
        self._vad_framer.start_segment(start)
        self._vad_policy.start_segment(start)
        self._wake_policy.start_segment(start, generation=generation)
        self._activation.start_segment(start)
        self._generation = generation
        self._segment_active = True
        self._diagnostic_vad_speech = False
        self._final_vad_ranges = None
        self._vad_finalization_attempted = False
        self._final_vad_consumed = 0
        self._wake_finalized = False
        self._segment_output_assembled = False
        self._activation_finalized = False

    def _finish_segment(self, wake_reason: WakeCandidateEndReason) -> list[OutputChunk]:
        end = self._timeline.next_position
        if self._final_vad_ranges is None:
            self._vad_finalization_attempted = True
            self._final_vad_ranges = self._vad_policy.finish(end)
        while self._final_vad_consumed < len(self._final_vad_ranges):
            vad_range = self._final_vad_ranges[self._final_vad_consumed]
            self._final_vad_consumed += 1
            self._consume_vad(
                [vad_range],
                reason=VadTransitionReason.SEGMENT_END,
            )
        if not self._wake_finalized:
            if self._wake_policy.evaluating:
                self._logger.info("wake_candidate_rejected", position=end)
            try:
                self._wake_policy.finish(end, reason=wake_reason)
            finally:
                self._wake_finalized = True
        if self._diagnostics is not None and self._diagnostic_vad_speech:
            assert self._generation is not None
            self._diagnostics.append(VadSpeechDiagnostic(self._generation, end, False, VadTransitionReason.SEGMENT_END))
        outputs: list[OutputChunk] = []
        if not self._segment_output_assembled:
            outputs = self._emit_through(end, evaluating=False)
            self._segment_output_assembled = True
        if not self._activation_finalized:
            try:
                self._activation.finish_segment(end)
            except Exception:
                self._failure_outputs.extend(outputs)
                raise
            finally:
                self._activation_finalized = True
        self._segment_active = False
        self._generation = None
        return outputs

    def _advance(self) -> list[OutputChunk]:
        results = self._vad_framer.consume(self._timeline, self._timeline.next_position)
        if self._diagnostics is not None:
            assert self._generation is not None
            for result in results:
                self._diagnostics.append(
                    VadFrameDiagnostic(
                        self._generation,
                        result.sample_range.start,
                        result.sample_range.end,
                        result.value,
                    )
                )
        self._consume_vad(self._vad_policy.consume(results))
        return self._emit_through(self._wake_policy.safe_frontier or self._timeline.retained_range.start)

    def _consume_vad(
        self,
        ranges: list[VadRange],
        *,
        reason: VadTransitionReason = VadTransitionReason.EVIDENCE,
    ) -> None:
        self._record_vad_ranges(ranges, reason)
        self._pending_vad.extend(ranges)
        for vad_range in ranges:
            update = self._wake_policy.consume(self._timeline, [vad_range])
            if update.decision is not None:
                self._pending_decision = update.decision
                self._wake_policy.activate()
                if self._diagnostics is not None:
                    assert self._generation is not None
                    decision = update.decision
                    self._diagnostics.append(
                        ActivationDiagnostic(
                            self._generation,
                            decision.activation_start,
                            decision.first_score_range.start,
                            decision.first_score_range.end,
                            decision.first_score,
                            decision.confirming_score_range.start,
                            decision.confirming_score_range.end,
                            decision.confirming_score,
                        )
                    )
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
            self._timeline.release_before(ranges[-1].sample_range.end)
        except Exception:
            self._pending_vad = pending_before
            self._pending_decision = decision_before
            self._activation = activation_before
            raise
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

    def _record_vad_ranges(self, ranges: list[VadRange], reason: VadTransitionReason) -> None:
        if self._diagnostics is None:
            return
        assert self._generation is not None
        for vad_range in ranges:
            if vad_range.is_speech != self._diagnostic_vad_speech:
                self._diagnostics.append(
                    VadSpeechDiagnostic(
                        self._generation,
                        vad_range.sample_range.start,
                        vad_range.is_speech,
                        reason,
                    )
                )
                self._diagnostic_vad_speech = vad_range.is_speech

    def _record_boundary(self, previous: int | None, next_generation: int, discontinuity: bool) -> None:
        if self._diagnostics is None:
            return
        generation_changed = previous is not None and previous != next_generation
        if generation_changed and discontinuity:
            reason = ContinuityBoundaryReason.GENERATION_CHANGE_AND_DISCONTINUITY
        elif generation_changed:
            reason = ContinuityBoundaryReason.GENERATION_CHANGE
        else:
            reason = ContinuityBoundaryReason.INPUT_DISCONTINUITY
        self._diagnostics.append(
            ContinuityBoundaryDiagnostic(
                self._timeline.next_position,
                previous,
                next_generation,
                reason,
            )
        )

    def _require_open(self) -> None:
        if self._closed or self._finished or self._failed:
            raise StreamStateError("processor is closed or finished")

    def _enter_failure(
        self,
        error: BaseException,
        outputs: list[OutputChunk],
        operation: DiagnosticFailureOperation,
        *,
        tail: InputChunk | None = None,
    ) -> None:
        self._failed = True
        self._failure_outputs.extend(outputs)
        if tail is not None:
            self._failure_tail = tail
        self._failure_tail_is_speech = self._vad_policy.failure_is_speech if self._segment_active else False
        with suppress(Exception):
            self._logger.error(
                "stream_processing_failed",
                generation=self._generation,
                position=self._timeline.next_position,
                pending_samples=self._timeline.retained_samples,
                error=str(error),
            )
            if self._diagnostics is not None:
                self._diagnostics.append(
                    FailureDiagnostic(
                        self._generation,
                        self._timeline.next_position,
                        self._accepted_samples,
                        operation,
                        type(error).__name__,
                        str(error),
                    )
                )

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
