from __future__ import annotations

import numpy as np
import pytest

from lumivox_wakelab import (
    InputChunk,
    StreamConfig,
    VadPolicyConfig,
    WakePolicyConfig,
    VadFrameDiagnostic,
    WakeFrameDiagnostic,
    WakeStreamProcessor,
    ActivationDiagnostic,
    ContinuityBoundaryReason,
    ContinuityBoundaryDiagnostic,
    WakeCandidateEndedDiagnostic,
    WakeCandidateStartedDiagnostic,
)

from .test_processor import FakeBackend, RecordingLogger, chunk, config, canonical


def processor(capacity: int) -> WakeStreamProcessor:
    return WakeStreamProcessor(
        config(),
        FakeBackend([0.9, 0.9, 0.9]),
        FakeBackend([0.0, 0.9]),
        logger=RecordingLogger(),
        diagnostic_event_capacity=capacity,
    )


def test_diagnostics_are_disabled_by_default_and_do_not_change_output() -> None:
    disabled = processor(0)
    enabled = processor(64)

    disabled_output = disabled.process(chunk(list(range(12)))) + disabled.finish()
    enabled_output = enabled.process(chunk(list(range(12)))) + enabled.finish()

    assert canonical(disabled_output) == canonical(enabled_output)
    assert disabled.drain_diagnostics().events == ()
    assert enabled.drain_diagnostics().events


def test_diagnostics_report_exact_frames_streak_and_activation_scores() -> None:
    stream = processor(64)
    stream.process(chunk(list(range(12))))
    events = stream.drain_diagnostics().events

    vad_frames = [event for event in events if isinstance(event, VadFrameDiagnostic)]
    wake_frames = [event for event in events if isinstance(event, WakeFrameDiagnostic)]
    assert [(event.start_sample, event.end_sample, event.probability) for event in vad_frames] == [
        (0, 4, 0.9),
        (4, 8, 0.9),
        (8, 12, 0.9),
    ]
    assert [
        (event.start_sample, event.end_sample, event.score, event.consecutive_hit_count) for event in wake_frames
    ] == [
        (0, 4, 0.0, 0),
        (4, 8, 0.9, 1),
    ]
    assert any(isinstance(event, WakeCandidateStartedDiagnostic) for event in events)
    assert any(isinstance(event, WakeCandidateEndedDiagnostic) for event in events)
    activation = next(event for event in events if isinstance(event, ActivationDiagnostic))
    assert (activation.activation_start_sample, activation.first_score, activation.confirming_score) == (2, 0.9, 0.9)


def test_diagnostic_overflow_drops_oldest_and_resets_count_on_drain() -> None:
    stream = processor(2)
    stream.process(chunk(list(range(12))))

    first = stream.drain_diagnostics()
    assert len(first.events) == 2
    assert first.dropped_events > 0
    assert stream.drain_diagnostics().dropped_events == 0


def test_boundaries_and_rearm_are_observable_between_segments() -> None:
    stream = processor(64)
    stream.process(chunk([0, 1, 2, 3]))
    stream.discontinue()
    stream.rearm()
    output = stream.process(chunk([4, 5, 6, 7], running_time_ns=1_000))
    output.extend(stream.finish())

    boundaries = [
        event for event in stream.drain_diagnostics().events if isinstance(event, ContinuityBoundaryDiagnostic)
    ]
    assert len(boundaries) == 1
    assert boundaries[0].reason is ContinuityBoundaryReason.EXPLICIT_DISCONTINUE
    assert canonical(output)[0][-1]


def test_rearm_between_segments_clears_latch_for_next_audio() -> None:
    stream = WakeStreamProcessor(
        config(),
        FakeBackend([0.9, 0.9, 0.9, 0.9]),
        FakeBackend([0.9, 0.0, 0.0]),
        logger=RecordingLogger(),
        diagnostic_event_capacity=64,
    )
    first = stream.process(chunk(list(range(8))))
    first.extend(stream.discontinue())
    stream.rearm()
    second = stream.process(chunk(list(range(8, 16))))
    second.extend(stream.finish())

    assert any(item.is_activated for item in first)
    assert all(not item.is_activated for item in second)


@pytest.mark.parametrize("capacity", [-1, True, 1.5])
def test_diagnostic_capacity_is_validated(capacity: object) -> None:
    error = TypeError if isinstance(capacity, (bool, float)) else ValueError
    with pytest.raises(error):
        WakeStreamProcessor(
            StreamConfig(
                VadPolicyConfig(0.6, 0.4, 4, 4, 0, 0),
                WakePolicyConfig(0.7, 1, 4, 2),
                16,
            ),
            FakeBackend([]),
            FakeBackend([]),
            logger=RecordingLogger(),
            diagnostic_event_capacity=capacity,  # type: ignore[arg-type]
        )


def test_diagnostic_drain_retains_no_pcm() -> None:
    stream = processor(64)
    samples = np.arange(12, dtype=np.dtype("<i2"))
    stream.process(InputChunk(samples, 0, 0, 0, False))

    assert all(not hasattr(event, "samples") for event in stream.drain_diagnostics().events)
