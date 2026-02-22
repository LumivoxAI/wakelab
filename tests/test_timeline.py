from __future__ import annotations

import numpy as np
import pytest

from lumivox_wakelab.stream import InputChunk
from lumivox_wakelab._timeline import SampleRange, AudioTimeline


def chunk(
    values: list[int],
    *,
    running_time_ns: int = 0,
    captured_at_ns: int = 0,
    generation: int = 0,
    discontinuity: bool = False,
) -> InputChunk:
    return InputChunk(
        samples=np.array(values, dtype=np.dtype("<i2")),
        running_time_ns=running_time_ns,
        captured_at_ns=captured_at_ns,
        generation=generation,
        discontinuity=discontinuity,
    )


def test_append_read_release_and_wraparound_preserve_positions() -> None:
    timeline = AudioTimeline(5)

    assert timeline.append(chunk([0, 1, 2])) == SampleRange(0, 3)
    timeline.release_before(2)
    assert timeline.append(chunk([3, 4, 5, 6])) == SampleRange(3, 7)

    assert timeline.retained_range == SampleRange(2, 7)
    np.testing.assert_array_equal(timeline.read(2, 7), [2, 3, 4, 5, 6])


def test_append_overflow_is_atomic() -> None:
    timeline = AudioTimeline(3)
    timeline.append(chunk([1, 2]))

    with pytest.raises(BufferError):
        timeline.append(chunk([3, 4]))

    assert timeline.retained_range == SampleRange(0, 2)
    np.testing.assert_array_equal(timeline.read(0, 2), [1, 2])


def test_storage_and_reads_do_not_alias_caller_or_each_other() -> None:
    samples = np.array([1, 2], dtype=np.dtype("<i2"))
    timeline = AudioTimeline(2)
    timeline.append(
        InputChunk(
            samples=samples,
            running_time_ns=0,
            captured_at_ns=0,
            generation=0,
            discontinuity=False,
        )
    )
    samples[0] = 9

    first = timeline.read(0, 2)
    first[0] = 8

    np.testing.assert_array_equal(timeline.read(0, 2), [1, 2])


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_must_be_a_positive_integer(capacity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AudioTimeline(capacity)  # type: ignore[arg-type]


def test_range_and_release_validation() -> None:
    timeline = AudioTimeline(4)
    timeline.append(chunk([1, 2]))

    with pytest.raises(ValueError):
        timeline.read(1, 1)
    with pytest.raises(ValueError):
        timeline.read(0, 3)
    with pytest.raises(ValueError):
        timeline.release_before(3)


def test_timestamps_use_nearest_anchor_without_cumulative_drift() -> None:
    timeline = AudioTimeline(8)
    timeline.append(chunk([0, 1, 2], running_time_ns=1, captured_at_ns=11))
    timeline.append(chunk([3, 4], running_time_ns=187_501, captured_at_ns=187_511))

    metadata = timeline.metadata_at(4)

    assert metadata.running_time_ns == 250_001
    assert metadata.captured_at_ns == 250_011
    assert metadata.generation == 0
    assert not metadata.discontinuity


def test_one_sample_running_time_discrepancy_is_accepted() -> None:
    timeline = AudioTimeline(4)
    timeline.append(chunk([0, 1], running_time_ns=10))
    timeline.append(chunk([2], running_time_ns=135_010))

    with pytest.raises(ValueError, match="disagrees"):
        timeline.append(chunk([3], running_time_ns=260_011))


def test_timed_and_untimed_modes_cannot_mix_within_a_segment() -> None:
    timeline = AudioTimeline(4)
    timeline.append(chunk([0], running_time_ns=0))
    with pytest.raises(ValueError, match="cannot become available"):
        timeline.append(chunk([1], running_time_ns=1))

    timed = AudioTimeline(4)
    timed.append(chunk([0], running_time_ns=1))
    with pytest.raises(ValueError, match="cannot become unavailable"):
        timed.append(chunk([1], running_time_ns=0))


def test_new_segments_reset_timing_validation_and_mark_their_first_output() -> None:
    timeline = AudioTimeline(8)
    timeline.append(chunk([0, 1], running_time_ns=1, generation=1))
    timeline.append(chunk([2], running_time_ns=0, generation=2))
    timeline.append(chunk([3], running_time_ns=0, generation=2))
    timeline.append(chunk([4], running_time_ns=999_999, generation=2, discontinuity=True))

    inferred = timeline.metadata_at(2)
    explicit = timeline.metadata_at(4)
    assert inferred.discontinuity
    assert inferred.running_time_ns == 0
    assert inferred.generation == 2
    assert explicit.discontinuity
    assert explicit.running_time_ns == 999_999


def test_release_preserves_the_nearest_preceding_anchor() -> None:
    timeline = AudioTimeline(8)
    timeline.append(chunk([0, 1, 2], running_time_ns=10))
    timeline.append(chunk([3, 4], running_time_ns=187_510))
    timeline.release_before(1)

    assert timeline.metadata_at(1).running_time_ns == 62_510
    assert timeline.metadata_at(4).running_time_ns == 250_010
