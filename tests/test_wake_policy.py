from __future__ import annotations

import numpy as np
import pytest

from lumivox_wakelab.stream import InputChunk, WakePolicyConfig
from lumivox_wakelab._framing import WakeWordFramer
from lumivox_wakelab._timeline import SampleRange, AudioTimeline
from lumivox_wakelab.vad._policy import VadRange
from lumivox_wakelab.wakeword._policy import WakePolicy, WakeDecision


class FakeBackend:
    sample_rate = 16_000
    frame_samples = 4

    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)
        self.frames: list[list[int]] = []
        self.reset_calls = 0

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        self.frames.append(frame.tolist())
        return next(self._scores)

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        pass


def make_policy(scores: list[float], **overrides: int | float) -> tuple[WakePolicy, AudioTimeline, FakeBackend]:
    config = WakePolicyConfig(
        score_threshold=float(overrides.get("score_threshold", 0.5)),
        consecutive_score_count=int(overrides.get("consecutive_score_count", 2)),
        silence_bridge_samples=int(overrides.get("silence_bridge_samples", 4)),
        pre_roll_samples=int(overrides.get("pre_roll_samples", 3)),
    )
    backend = FakeBackend(scores)
    policy = WakePolicy(config, WakeWordFramer(backend))
    timeline = AudioTimeline(64)
    timeline.append(InputChunk(np.arange(32, dtype=np.dtype("<i2")), 0, 0, 0, False))
    policy.start_segment(0)
    return policy, timeline, backend


def ranges(*values: tuple[int, int, bool]) -> list[VadRange]:
    return [VadRange(SampleRange(start, end), is_speech) for start, end, is_speech in values]


def test_armed_silence_skips_wake_inference_and_is_immediately_safe() -> None:
    policy, timeline, backend = make_policy([])

    update = policy.consume(timeline, ranges((0, 8, False)))

    assert update.decision is None
    assert update.safe_frontier == 8
    assert backend.frames == []


@pytest.mark.parametrize(
    ("gap_end", "speech_end", "expected_frames"),
    [
        (7, 8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (8, 12, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
    ],
)
def test_bridge_limit_is_inclusive_and_gap_audio_is_contiguous(
    gap_end: int, speech_end: int, expected_frames: list[list[int]]
) -> None:
    policy, timeline, backend = make_policy([0.0, 0.0, 0.0], silence_bridge_samples=4)

    policy.consume(timeline, ranges((0, 4, True), (4, gap_end, False), (gap_end, speech_end, True)))

    assert backend.frames == expected_frames
    assert policy.evaluating


def test_gap_longer_than_limit_infers_complete_frames_before_rejecting() -> None:
    policy, timeline, backend = make_policy([0.0, 0.0], silence_bridge_samples=4)

    update = policy.consume(timeline, ranges((0, 4, True), (4, 9, False), (9, 12, True)))

    assert backend.frames == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert update.safe_frontier == 9
    assert policy.evaluating


def test_confirming_score_inside_long_gap_activates_before_bridge_rejection() -> None:
    policy, timeline, backend = make_policy([0.9, 0.9], silence_bridge_samples=4)

    update = policy.consume(timeline, ranges((0, 4, True), (4, 12, False)))

    assert update.decision == WakeDecision(0, SampleRange(0, 4), SampleRange(4, 8), 0.9, 0.9)
    assert backend.frames == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_bridge_detection_is_independent_of_non_speech_partition() -> None:
    decisions: list[WakeDecision | None] = []
    for non_speech in (((4, 12, False),), ((4, 6, False), (6, 12, False))):
        policy, timeline, _ = make_policy([0.9, 0.9], silence_bridge_samples=4)
        update = policy.consume(timeline, ranges((0, 4, True), *non_speech))
        decisions.append(update.decision)

    assert decisions[0] == decisions[1]
    assert decisions[0] is not None


def test_score_confirmation_anchors_pre_roll_to_first_qualifying_frame() -> None:
    policy, timeline, _ = make_policy([0.4, 0.5, 0.9], pre_roll_samples=3)

    update = policy.consume(timeline, ranges((0, 12, True)))

    assert update.decision == WakeDecision(1, SampleRange(4, 8), SampleRange(8, 12), 0.5, 0.9)
    assert update.safe_frontier == 1


def test_lower_score_breaks_streak_and_frontier_advances_after_it() -> None:
    policy, timeline, _ = make_policy([0.9, 0.4, 0.9], pre_roll_samples=3)

    update = policy.consume(timeline, ranges((0, 12, True)))

    assert update.decision is None
    assert update.safe_frontier == 5


def test_decision_is_independent_of_vad_range_partition() -> None:
    expected: WakeDecision | None = None
    for partition in ([12], [4, 8], [4, 4, 4]):
        policy, timeline, _ = make_policy([0.4, 0.5, 0.9], pre_roll_samples=3)
        start = 0
        decision: WakeDecision | None = None
        for length in partition:
            update = policy.consume(timeline, ranges((start, start + length, True)))
            decision = update.decision or decision
            start += length
        if expected is None:
            expected = decision
        assert decision == expected


def test_one_score_is_enough_and_activation_bypasses_later_inference() -> None:
    policy, timeline, backend = make_policy([0.5], consecutive_score_count=1)

    update = policy.consume(timeline, ranges((0, 4, True)))
    assert update.decision is not None
    policy.activate()
    later = policy.consume(timeline, ranges((4, 12, True)))

    assert backend.frames == [[0, 1, 2, 3]]
    assert later.safe_frontier == 12


def test_rearm_and_finish_reject_unconfirmed_hits() -> None:
    policy, timeline, backend = make_policy([0.9, 0.9], consecutive_score_count=2)
    policy.consume(timeline, ranges((0, 4, True)))

    policy.rearm()
    update = policy.consume(timeline, ranges((4, 8, True)))
    finished = policy.finish(10)

    assert update.decision is None
    assert backend.reset_calls == 4
    assert finished.safe_frontier == 10


def test_ranges_must_be_contiguous_and_retained() -> None:
    policy, timeline, _ = make_policy([])

    with pytest.raises(ValueError, match="contiguous"):
        policy.consume(timeline, ranges((1, 4, False)))
    with pytest.raises(ValueError, match="retained"):
        policy.consume(timeline, ranges((0, 40, False)))
