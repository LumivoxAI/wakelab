"""Backend-independent candidate and score policy for wake-word detection."""

from __future__ import annotations

from dataclasses import dataclass

from lumivox_wakelab.stream import WakePolicyConfig
from lumivox_wakelab._framing import FrameResult, WakeWordFramer
from lumivox_wakelab._timeline import SampleRange, AudioTimeline
from lumivox_wakelab.diagnostics import (
    WakeFrameDiagnostic,
    WakeCandidateEndReason,
    WakeCandidateEndedDiagnostic,
    WakeCandidateStartedDiagnostic,
)
from lumivox_wakelab.vad._policy import VadRange
from lumivox_wakelab._diagnostics import DiagnosticCollector


@dataclass(frozen=True, slots=True)
class WakeDecision:
    """A confirmed wake score sequence and its bounded activation start."""

    activation_start: int
    first_score_range: SampleRange
    confirming_score_range: SampleRange
    first_score: float = 0.0
    confirming_score: float = 0.0


@dataclass(frozen=True, slots=True)
class WakePolicyUpdate:
    """One policy-step result for the stream orchestrator."""

    decision: WakeDecision | None
    safe_frontier: int


class WakePolicy:
    """Gate wake inference by finalized speech and bound revisable audio."""

    def __init__(
        self,
        config: WakePolicyConfig,
        framer: WakeWordFramer,
        /,
        *,
        diagnostics: DiagnosticCollector | None = None,
    ) -> None:
        if not isinstance(config, WakePolicyConfig):
            raise TypeError("config must be a WakePolicyConfig")
        if not isinstance(framer, WakeWordFramer):
            raise TypeError("framer must be a WakeWordFramer")
        self._config = config
        self._framer = framer
        self._segment_start: int | None = None
        self._position: int | None = None
        self._candidate_start: int | None = None
        self._gap_start: int | None = None
        self._first_hit: FrameResult | None = None
        self._hit_count = 0
        self._activated = False
        self._generation: int | None = None
        self._armed_from = 0
        self._diagnostics = diagnostics

    @property
    def safe_frontier(self) -> int | None:
        """Exclusive position before which activation is no longer possible."""

        if self._position is None:
            return None
        if self._activated or self._candidate_start is None:
            return self._position
        assert self._framer.position is not None
        earliest = self._first_hit.sample_range.start if self._first_hit is not None else self._framer.position
        return max(self._candidate_start, earliest - self._config.pre_roll_samples)

    @property
    def evaluating(self) -> bool:
        return self._candidate_start is not None

    @property
    def activated(self) -> bool:
        return self._activated

    def start_segment(self, start: int, /, *, generation: int = 0) -> None:
        """Begin a continuity segment after resetting transient candidate state."""

        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("segment start must be an integer")
        if start < 0:
            raise ValueError("segment start must be nonnegative")
        if self._position is not None:
            raise RuntimeError("wake-policy segment is already active")
        self._framer.reset()
        self._segment_start = start
        self._position = start
        self._candidate_start = None
        self._gap_start = None
        self._clear_hits()
        self._generation = generation
        self._armed_from = start

    def consume(self, timeline: AudioTimeline, ranges: list[VadRange], /) -> WakePolicyUpdate:
        """Process contiguous finalized VAD ranges retained in ``timeline``."""

        if self._position is None:
            raise RuntimeError("wake-policy segment has not been started")
        self._validate_ranges(timeline, ranges)
        decision: WakeDecision | None = None
        for vad_range in ranges:
            sample_range = vad_range.sample_range
            if self._activated:
                self._position = sample_range.end
                continue
            if vad_range.is_speech:
                if self._candidate_start is None:
                    candidate_start = max(sample_range.start, self._armed_from)
                    if candidate_start >= sample_range.end:
                        self._position = sample_range.end
                        continue
                    self._position = candidate_start
                    self._start_candidate(candidate_start)
                self._gap_start = None
                found = self._consume_through(timeline, sample_range.end)
                if found is not None:
                    decision = found
                    self._position = sample_range.end
                    break
                self._position = sample_range.end
            elif self._candidate_start is not None:
                self._position = sample_range.end
                self._handle_gap(sample_range)
            else:
                self._position = sample_range.end

        assert self.safe_frontier is not None
        return WakePolicyUpdate(decision, self.safe_frontier)

    def activate(self) -> None:
        """Latch policy bypass after the caller applies a confirmed decision."""

        if self._position is None:
            raise RuntimeError("wake-policy segment has not been started")
        self._activated = True
        self._end_candidate(WakeCandidateEndReason.ACTIVATED)

    def rearm(self, boundary: int | None = None) -> None:
        """Cancel evaluation and resume score processing at the next speech range."""

        self._activated = False
        if self._position is not None:
            self._end_candidate(WakeCandidateEndReason.REARMED)
            self._armed_from = self._position if boundary is None else boundary
        self._framer.reset()

    def finish(
        self,
        end: int,
        /,
        *,
        reason: WakeCandidateEndReason = WakeCandidateEndReason.FINISHED,
    ) -> WakePolicyUpdate:
        """Reject unresolved evaluation without inferring an incomplete final frame."""

        if self._position is None:
            raise RuntimeError("wake-policy segment has not been started")
        if isinstance(end, bool) or not isinstance(end, int):
            raise TypeError("segment end must be an integer")
        if end < self._position:
            raise ValueError("segment end precedes finalized VAD ranges")
        self._position = end
        if not self._activated:
            self._end_candidate(reason)
        frontier = self.safe_frontier
        assert frontier is not None
        self._position = None
        self._segment_start = None
        self._generation = None
        return WakePolicyUpdate(None, frontier)

    def _start_candidate(self, start: int) -> None:
        self._framer.start_candidate(start)
        self._candidate_start = start
        self._gap_start = None
        self._clear_hits()
        if self._diagnostics is not None:
            assert self._generation is not None
            self._diagnostics.append(WakeCandidateStartedDiagnostic(self._generation, start))

    def _handle_gap(self, sample_range: SampleRange) -> None:
        if self._gap_start is None:
            self._gap_start = sample_range.start
        assert self._gap_start is not None
        if sample_range.end - self._gap_start > self._config.silence_bridge_samples:
            self._end_candidate(WakeCandidateEndReason.SILENCE_BRIDGE_EXCEEDED)

    def _consume_through(self, timeline: AudioTimeline, end: int) -> WakeDecision | None:
        while (result := self._framer.consume_next(timeline, end)) is not None:
            self._position = result.sample_range.end
            if result.value >= self._config.score_threshold:
                if self._first_hit is None:
                    self._first_hit = result
                self._hit_count += 1
                threshold_met = True
                if self._hit_count == self._config.consecutive_score_count:
                    first = self._first_hit
                    assert first is not None and self._candidate_start is not None
                    decision = WakeDecision(
                        activation_start=max(
                            self._candidate_start,
                            first.sample_range.start - self._config.pre_roll_samples,
                        ),
                        first_score_range=first.sample_range,
                        first_score=first.value,
                        confirming_score_range=result.sample_range,
                        confirming_score=result.value,
                    )
                else:
                    decision = None
            else:
                self._clear_hits()
                threshold_met = False
                decision = None
            if self._diagnostics is not None:
                assert self._generation is not None
                self._diagnostics.append(
                    WakeFrameDiagnostic(
                        self._generation,
                        result.sample_range.start,
                        result.sample_range.end,
                        result.value,
                        threshold_met,
                        self._hit_count,
                    )
                )
            if decision is not None:
                return decision
        return None

    def _end_candidate(self, reason: WakeCandidateEndReason) -> None:
        if self._candidate_start is not None:
            assert self._position is not None
            start = self._candidate_start
            self._framer.end_candidate(self._position)
            if self._diagnostics is not None:
                assert self._generation is not None
                self._diagnostics.append(WakeCandidateEndedDiagnostic(self._generation, start, self._position, reason))
        self._candidate_start = None
        self._gap_start = None
        self._clear_hits()

    def _clear_hits(self) -> None:
        self._first_hit = None
        self._hit_count = 0

    def _validate_ranges(self, timeline: AudioTimeline, ranges: list[VadRange]) -> None:
        assert self._position is not None
        position = self._position
        retained = timeline.retained_range
        for vad_range in ranges:
            if not isinstance(vad_range, VadRange):
                raise TypeError("ranges must contain VadRange values")
            sample_range = vad_range.sample_range
            if sample_range.start != position or sample_range.end <= position:
                raise ValueError("VAD ranges must be contiguous nonempty ranges")
            if sample_range.start < retained.start or sample_range.end > retained.end:
                raise ValueError("VAD ranges must be retained")
            if not isinstance(vad_range.is_speech, bool):
                raise TypeError("VadRange.is_speech must be a bool")
            position = sample_range.end
