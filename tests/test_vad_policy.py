from __future__ import annotations

import pytest

from lumivox_wakelab.stream import VadPolicyConfig
from lumivox_wakelab._framing import FrameResult
from lumivox_wakelab._timeline import SampleRange
from lumivox_wakelab.vad._policy import VadRange, VadPolicy


def frames(values: list[float], frame_samples: int = 4, start: int = 0) -> list[FrameResult]:
    return [
        FrameResult(
            SampleRange(start + index * frame_samples, start + (index + 1) * frame_samples),
            value,
        )
        for index, value in enumerate(values)
    ]


def merged(ranges: list[VadRange]) -> list[VadRange]:
    result: list[VadRange] = []
    for current in ranges:
        if (
            result
            and result[-1].is_speech == current.is_speech
            and result[-1].sample_range.end == current.sample_range.start
        ):
            previous = result[-1]
            result[-1] = VadRange(SampleRange(previous.sample_range.start, current.sample_range.end), current.is_speech)
        else:
            result.append(current)
    return result


def policy(*, left_padding_samples: int = 0, right_padding_samples: int = 0) -> VadPolicy:
    config = VadPolicyConfig(
        speech_threshold=0.6,
        silence_threshold=0.4,
        minimum_speech_samples=8,
        minimum_silence_samples=8,
        left_padding_samples=left_padding_samples,
        right_padding_samples=right_padding_samples,
    )
    return VadPolicy(config)


@pytest.mark.parametrize("minimum", [1, 4, 5, 8, 12])
@pytest.mark.parametrize("speech", [False, True])
def test_transition_confirms_on_the_first_frame_reaching_sample_minimum(minimum: int, speech: bool) -> None:
    config = VadPolicyConfig(0.6, 0.4, minimum, minimum, 0, 0)
    vad = VadPolicy(config)
    vad.start_segment(0)
    values = [0.9] * ((minimum + 3) // 4)
    if not speech:
        vad.consume(frames(values))
        values = [0.1] * ((minimum + 3) // 4)
    output = vad.consume(frames(values, start=vad.finalized_position or 0))
    end = ((minimum + 3) // 4) * 4 * (2 if not speech else 1)
    output.extend(vad.finish(end))

    expected = speech if speech else False
    assert merged(output)[-1].is_speech is expected


def test_hysteresis_thresholds_and_exact_speech_confirmation() -> None:
    vad = policy(left_padding_samples=2)
    vad.start_segment(0)

    assert vad.consume(frames([0.4, 0.6])) == [VadRange(SampleRange(0, 2), False)]
    assert vad.consume(frames([0.5], start=8)) == [VadRange(SampleRange(2, 12), True)]
    assert vad.finish(12) == []


def test_rejected_onset_and_stable_silence_retain_only_left_padding() -> None:
    vad = policy(left_padding_samples=3)
    vad.start_segment(0)

    assert vad.consume(frames([0.1, 0.1])) == [VadRange(SampleRange(0, 5), False)]
    assert vad.consume(frames([0.9, 0.1], start=8)) == [VadRange(SampleRange(5, 13), False)]
    assert vad.finish(16) == [VadRange(SampleRange(13, 16), False)]


def test_short_silence_is_bridged_and_confirmed_silence_applies_padding() -> None:
    vad = policy(right_padding_samples=2)
    vad.start_segment(0)

    assert vad.consume(frames([0.9, 0.9])) == [VadRange(SampleRange(0, 8), True)]
    assert vad.consume(frames([0.1, 0.9], start=8)) == [VadRange(SampleRange(8, 16), True)]
    assert vad.consume(frames([0.1, 0.1], start=16)) == [
        VadRange(SampleRange(16, 18), True),
        VadRange(SampleRange(18, 24), False),
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.9], [VadRange(SampleRange(0, 4), False)]),
        ([0.9, 0.9, 0.1], [VadRange(SampleRange(0, 12), True)]),
    ],
)
def test_finish_rejects_unconfirmed_onset_and_keeps_confirmed_speech(
    values: list[float], expected: list[VadRange]
) -> None:
    vad = policy()
    vad.start_segment(0)
    output = vad.consume(frames(values))

    output.extend(vad.finish(len(values) * 4))
    assert merged(output) == expected


def test_incomplete_tail_uses_last_committed_state() -> None:
    vad = policy()
    vad.start_segment(0)
    assert vad.consume(frames([0.9, 0.9])) == [VadRange(SampleRange(0, 8), True)]

    assert vad.finish(11) == [VadRange(SampleRange(8, 11), True)]


def test_results_must_be_contiguous_and_probabilities_valid() -> None:
    vad = policy()
    vad.start_segment(0)

    with pytest.raises(ValueError, match="contiguous"):
        vad.consume([FrameResult(SampleRange(1, 5), 0.5)])
    with pytest.raises(ValueError, match="between"):
        vad.consume([FrameResult(SampleRange(0, 4), 1.1)])


def test_policy_output_is_independent_of_result_partition() -> None:
    expected: list[VadRange] | None = None
    source = frames([0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    for partition in ([6], [1, 2, 3], [2, 2, 2]):
        vad = policy(left_padding_samples=2, right_padding_samples=2)
        vad.start_segment(0)
        output: list[VadRange] = []
        offset = 0
        for length in partition:
            output.extend(vad.consume(source[offset : offset + length]))
            offset += length
        output.extend(vad.finish(24))
        if expected is None:
            expected = output
        assert merged(output) == merged(expected)
