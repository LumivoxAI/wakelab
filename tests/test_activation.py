from __future__ import annotations

import pytest

from lumivox_wakelab._timeline import SampleRange
from lumivox_wakelab._activation import ActivationState, ActivationLifecycle
from lumivox_wakelab.vad._policy import VadRange
from lumivox_wakelab.wakeword._policy import WakeDecision


def decision(start: int) -> WakeDecision:
    return WakeDecision(start, SampleRange(start + 2, start + 4), SampleRange(start + 4, start + 6))


def ranges(*values: tuple[int, int, bool]) -> list[VadRange]:
    return [VadRange(SampleRange(start, end), is_speech) for start, end, is_speech in values]


def output_values(
    lifecycle: ActivationLifecycle,
    values: list[VadRange],
    wake: WakeDecision | None,
    *,
    evaluating: bool,
) -> list[tuple[int, int, bool, bool]]:
    return [
        (item.sample_range.start, item.sample_range.end, item.is_speech, item.is_activated)
        for item in lifecycle.consume(values, wake, evaluating=evaluating)
    ]


@pytest.mark.parametrize(
    ("activation_start", "expected"),
    [
        (0, [(0, 8, True, True)]),
        (3, [(0, 3, True, False), (3, 8, True, True)]),
        (8, [(0, 8, True, False)]),
    ],
)
def test_decision_splits_retained_audio_at_exact_activation_start(
    activation_start: int, expected: list[tuple[int, int, bool, bool]]
) -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)

    actual = output_values(lifecycle, ranges((0, 8, True)), decision(activation_start), evaluating=True)

    assert actual == expected
    assert lifecycle.state is ActivationState.ACTIVATED
    assert lifecycle.trigger == decision(activation_start)
    assert lifecycle.wake_inference_bypassed


def test_activation_and_speech_metadata_remain_independent() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)

    actual = output_values(
        lifecycle,
        ranges((0, 2, False), (2, 4, True), (4, 6, False)),
        decision(2),
        evaluating=True,
    )

    assert actual == [(0, 2, False, False), (2, 4, True, True), (4, 6, False, True)]


def test_activation_latches_across_silence_and_continuity_boundaries() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)
    output_values(lifecycle, ranges((0, 4, True)), decision(0), evaluating=True)
    later = output_values(lifecycle, ranges((4, 8, False)), None, evaluating=False)
    lifecycle.finish_segment(8)
    lifecycle.start_segment(8)
    after_boundary = output_values(lifecycle, ranges((8, 12, True)), None, evaluating=False)

    assert later == [(4, 8, False, True)]
    assert after_boundary == [(8, 12, True, True)]
    assert lifecycle.wake_inference_bypassed


def test_rearm_clears_latch_and_evaluation_at_next_unconsumed_boundary() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)
    output_values(lifecycle, ranges((0, 4, True)), decision(0), evaluating=True)

    lifecycle.rearm(4)
    lifecycle.rearm(4)
    after_rearm = output_values(lifecycle, ranges((4, 8, True)), None, evaluating=True)

    assert lifecycle.state is ActivationState.EVALUATING
    assert lifecycle.trigger is None
    assert not lifecycle.wake_inference_bypassed
    assert after_rearm == [(4, 8, True, False)]


def test_new_decision_after_scheduled_rearm_activates_from_new_boundary() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)
    output_values(lifecycle, ranges((0, 4, True)), decision(0), evaluating=True)
    lifecycle.rearm(8)

    actual = output_values(lifecycle, ranges((4, 12, True)), decision(8), evaluating=True)

    assert actual == [(4, 12, True, True)]
    assert lifecycle.state is ActivationState.ACTIVATED


def test_finish_rejects_unresolved_evaluation_and_preserves_confirmed_activation() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)
    output_values(lifecycle, ranges((0, 4, True)), None, evaluating=True)
    lifecycle.finish_segment(4)

    assert lifecycle.state.value == "armed"
    lifecycle.start_segment(4)
    output_values(lifecycle, ranges((4, 8, True)), decision(4), evaluating=True)
    lifecycle.finish_segment(8)

    assert lifecycle.state == ActivationState.ACTIVATED
    assert lifecycle.wake_inference_bypassed


def test_lifecycle_rejects_gaps_and_decisions_outside_retained_batch() -> None:
    lifecycle = ActivationLifecycle()
    lifecycle.start_segment(0)

    with pytest.raises(ValueError, match="contiguous"):
        lifecycle.consume(ranges((1, 4, True)), None, evaluating=False)
    with pytest.raises(ValueError, match="within"):
        lifecycle.consume(ranges((0, 4, True)), decision(5), evaluating=True)
