"""Backend-independent temporal policy for VAD frame probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lumivox_wakelab.stream import VadPolicyConfig
from lumivox_wakelab._framing import FrameResult
from lumivox_wakelab._timeline import SampleRange


@dataclass(frozen=True, slots=True)
class VadRange:
    """An immutable speech decision over one nonempty sample range."""

    sample_range: SampleRange
    is_speech: bool


class VadPolicy:
    """Finalize gap-free VAD ranges without depending on caller chunk boundaries."""

    def __init__(self, config: VadPolicyConfig, /) -> None:
        if not isinstance(config, VadPolicyConfig):
            raise TypeError("config must be a VadPolicyConfig")
        self._config = config
        self._segment_start: int | None = None
        self._position: int | None = None
        self._frontier: int | None = None
        self._state = "quiet"
        self._evidence_is_speech = False
        self._candidate_start: int | None = None
        self._candidate_samples = 0
        self._protected_speech_until: int | None = None

    @property
    def finalized_position(self) -> int | None:
        """Return the exclusive frontier of metadata that cannot be revised."""

        return self._frontier

    @property
    def failure_is_speech(self) -> bool:
        """Return the conservative classification for an unscored failure tail."""

        return self._state in {"speech", "silence"}

    def start_segment(self, start: int, /) -> None:
        """Begin a new continuity segment at an absolute sample position."""

        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("segment start must be an integer")
        if start < 0:
            raise ValueError("segment start must be nonnegative")
        if self._position is not None:
            raise RuntimeError("VAD segment is already active")
        self._segment_start = start
        self._position = start
        self._frontier = start
        self._state = "quiet"
        self._evidence_is_speech = False
        self._candidate_start = None
        self._candidate_samples = 0
        self._protected_speech_until = None

    def consume(self, results: list[FrameResult], /) -> list[VadRange]:
        """Apply contiguous complete-frame results and return finalized ranges."""

        if self._position is None:
            raise RuntimeError("VAD segment has not been started")
        self._validate_results(results)
        emitted: list[VadRange] = []
        for result in results:
            sample_range = result.sample_range
            self._position = sample_range.end
            self._update_evidence(result.value)
            if self._state == "quiet":
                if self._evidence_is_speech:
                    self._start_candidate(sample_range.start, "onset")
                else:
                    self._emit_quiet_through(sample_range.end, emitted)
            elif self._state == "onset":
                if self._evidence_is_speech:
                    self._candidate_samples += sample_range.end - sample_range.start
                    if self._candidate_samples >= self._config.minimum_speech_samples:
                        self._state = "speech"
                        self._emit_through(sample_range.end, True, emitted)
                else:
                    self._reject_candidate()
                    self._emit_quiet_through(sample_range.end, emitted)
            elif self._state == "speech":
                if self._evidence_is_speech:
                    self._emit_through(sample_range.end, True, emitted)
                else:
                    self._start_candidate(sample_range.start, "silence")
            else:  # self._state == "silence"
                if self._evidence_is_speech:
                    self._state = "speech"
                    self._candidate_start = None
                    self._candidate_samples = 0
                    self._emit_through(sample_range.end, True, emitted)
                else:
                    self._candidate_samples += sample_range.end - sample_range.start
                    if self._candidate_samples >= self._config.minimum_silence_samples:
                        assert self._candidate_start is not None
                        protected_until = self._candidate_start + self._config.right_padding_samples
                        self._protected_speech_until = max(self._protected_speech_until or 0, protected_until)
                        self._state = "quiet"
                        self._candidate_start = None
                        self._candidate_samples = 0
                        self._emit_quiet_through(sample_range.end, emitted)
        return emitted

    def finish(self, end: int, /) -> list[VadRange]:
        """Conservatively finalize real samples through a segment boundary or EOS."""

        if self._position is None or self._frontier is None:
            raise RuntimeError("VAD segment has not been started")
        if isinstance(end, bool) or not isinstance(end, int):
            raise TypeError("segment end must be an integer")
        if end < self._position:
            raise ValueError("segment end precedes processed VAD frames")

        emitted: list[VadRange] = []
        self._emit_through(end, self._state in {"speech", "silence"}, emitted)
        self._position = None
        self._segment_start = None
        self._candidate_start = None
        self._candidate_samples = 0
        return emitted

    def _validate_results(self, results: list[FrameResult]) -> None:
        assert self._position is not None
        position = self._position
        for result in results:
            if not isinstance(result, FrameResult):
                raise TypeError("results must contain FrameResult values")
            sample_range = result.sample_range
            if sample_range.start != position or sample_range.end <= position:
                raise ValueError("VAD frame results must be contiguous nonempty ranges")
            value = result.value
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("VAD probability must be a finite scalar")
            if not 0 <= value <= 1:
                raise ValueError("VAD probability must be between 0 and 1")
            position = sample_range.end

    def _update_evidence(self, value: float) -> None:
        if value >= self._config.speech_threshold:
            self._evidence_is_speech = True
        elif value < self._config.silence_threshold:
            self._evidence_is_speech = False

    def _start_candidate(self, start: int, state: str) -> None:
        self._state = state
        self._candidate_start = start
        self._candidate_samples = self._position - start if self._position is not None else 0

    def _reject_candidate(self) -> None:
        self._state = "quiet"
        self._candidate_start = None
        self._candidate_samples = 0

    def _emit_quiet_through(self, end: int, emitted: list[VadRange]) -> None:
        self._emit_through(max(self._frontier or 0, end - self._config.left_padding_samples), False, emitted)

    def _emit_through(self, end: int, is_speech: bool, emitted: list[VadRange]) -> None:
        assert self._frontier is not None
        if end < self._frontier:
            raise ValueError("finalization cannot move backwards")
        protected_until = self._protected_speech_until
        if not is_speech and protected_until is not None and self._frontier < protected_until:
            protected_end = min(end, protected_until)
            self._append_range(self._frontier, protected_end, True, emitted)
            self._frontier = protected_end
        self._append_range(self._frontier, end, is_speech, emitted)
        self._frontier = end

    @staticmethod
    def _append_range(start: int, end: int, is_speech: bool, emitted: list[VadRange]) -> None:
        if start >= end:
            return
        if emitted and emitted[-1].is_speech == is_speech and emitted[-1].sample_range.end == start:
            previous = emitted[-1]
            emitted[-1] = VadRange(SampleRange(previous.sample_range.start, end), is_speech)
            return
        emitted.append(VadRange(SampleRange(start, end), is_speech))
