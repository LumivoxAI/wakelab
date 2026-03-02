"""Synchronous assembly of framing, policies, and public output chunks."""

from __future__ import annotations

from collections import deque

from .stream import InputChunk, OutputChunk, StreamConfig, StreamStateError
from ._framing import VadFramer, WakeWordFramer
from ._timeline import SampleRange, AudioTimeline
from ._activation import ActivationRange, ActivationLifecycle
from .vad._policy import VadRange, VadPolicy
from .vad._backend import VadBackend
from .wakeword._policy import WakePolicy, WakeDecision
from .wakeword._backend import WakeWordBackend


class WakeStreamProcessor:
    """Process one PCM stream with injected VAD and wake-word backends."""

    def __init__(self, config: StreamConfig, vad_backend: VadBackend, wake_backend: WakeWordBackend, /) -> None:
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

    @property
    def accepted_samples(self) -> int:
        """Return the processor-lifetime count of accepted samples."""

        return self._timeline.next_position

    @property
    def emitted_samples(self) -> int:
        """Return the processor-lifetime count of emitted samples."""

        return self._timeline.retained_range.start

    @property
    def retained_samples(self) -> int:
        """Return the amount of currently undecided PCM."""

        return self._timeline.retained_samples

    def process(self, chunk: InputChunk, /) -> list[OutputChunk]:
        """Accept a valid chunk and return every range with final metadata."""

        if self._closed or self._finished:
            raise StreamStateError("processor is no longer accepting input")
        if not isinstance(chunk, InputChunk):
            raise TypeError("chunk must be an InputChunk")

        outputs: list[OutputChunk] = []
        if self._segment_active and (chunk.discontinuity or chunk.generation != self._generation):
            outputs.extend(self._finish_segment())
        if not self._segment_active:
            self._start_segment(self._timeline.next_position, chunk.generation)

        offset = 0
        while offset < chunk.samples.size:
            free = self._timeline.capacity - self._timeline.retained_samples
            if free == 0:
                outputs.extend(self._advance())
                free = self._timeline.capacity - self._timeline.retained_samples
                if free == 0:
                    raise BufferError("configured retained-audio bound cannot advance the stream")
            length = min(free, chunk.samples.size - offset)
            part = self._part(chunk, offset, length)
            self._timeline.append(part)
            offset += length
            outputs.extend(self._advance())
        return outputs

    def rearm(self) -> list[OutputChunk]:
        """Cancel wake evaluation and clear the activation latch for future audio."""

        self._require_open()
        if not self._segment_active:
            return []
        outputs = self._advance()
        # Pending VAD evidence is not yet public, so it belongs to the new armed state.
        self._wake_policy.rearm()
        self._activation.rearm(self._activation_position())
        return outputs

    def discontinue(self) -> list[OutputChunk]:
        """Finalize the current continuity segment without ending the processor."""

        self._require_open()
        return self._finish_segment() if self._segment_active else []

    def finish(self) -> list[OutputChunk]:
        """Finalize all real audio and permanently reject further input."""

        self._require_open()
        outputs = self._finish_segment() if self._segment_active else []
        self._finished = True
        return outputs

    def drain_failed(self) -> list[OutputChunk]:
        """Reserved for terminal-failure draining implemented by the next task."""

        raise StreamStateError("processor has not entered a terminal failure state")

    def close(self) -> list[OutputChunk]:
        """Finalize retained audio, close owned backends, and become terminal."""

        if self._closed:
            return []
        outputs = self.finish() if not self._finished else []
        self._vad_framer.close()
        self._wake_framer.close()
        self._closed = True
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
            update = self._wake_policy.consume(self._timeline, [vad_range])
            self._pending_vad.append(vad_range)
            if update.decision is not None:
                self._pending_decision = update.decision
                self._wake_policy.activate()

    def _emit_through(self, frontier: int) -> list[OutputChunk]:
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
        activated = self._activation.consume(ranges, decision, evaluating=self._wake_policy.evaluating)
        outputs = [self._output(item) for item in activated]
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
        if self._closed or self._finished:
            raise StreamStateError("processor is closed or finished")

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
