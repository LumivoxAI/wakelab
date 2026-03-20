# API Reference

This page covers the supported imports from `lumivox_wakelab`,
`lumivox_wakelab.vad`, and `lumivox_wakelab.wakeword`. Names in underscore-prefixed
modules are internal.

## Stream Values

All stream values are frozen, slotted dataclasses. NumPy arrays stored in chunks are
still writable.

### `InputChunk`

```python
InputChunk(samples, running_time_ns, captured_at_ns, generation, discontinuity)
```

| Field | Meaning |
| --- | --- |
| `samples` | Writable, nonempty, one-dimensional `numpy.ndarray` of exact dtype `numpy.dtype("<i2")`. |
| `running_time_ns` | Nonnegative stream-relative time of the first sample; zero means unavailable for this continuity segment. |
| `captured_at_ns` | Nonnegative wall-clock time anchor for the first sample. |
| `generation` | Nonnegative source generation. A change creates a continuity boundary. |
| `discontinuity` | Whether continuity with preceding input was lost. |

### `OutputChunk`

Contains the five input metadata dimensions plus `is_speech` and `is_activated`.
The two booleans are independent. Metadata applies uniformly to every sample in the
chunk. The first output at an input discontinuity, generation change, or explicit
`discontinue()` boundary has `discontinuity=True`; later output in that segment has
`False`.

## Policy Configuration

### `VadPolicyConfig`

| Field | Constraint and effect |
| --- | --- |
| `speech_threshold` | Finite probability in `[0, 1]`; a score at or above it is speech evidence. |
| `silence_threshold` | Finite probability in `[0, 1]`, strictly below `speech_threshold`; a score below it is silence evidence. |
| `minimum_speech_samples` | Positive sample count required to confirm speech. Confirmation is rounded to complete backend frames. |
| `minimum_silence_samples` | Positive sample count required to end speech. Confirmation is rounded to complete backend frames. |
| `left_padding_samples` | Nonnegative quiet pre-padding retained before confirmed speech. |
| `right_padding_samples` | Nonnegative padding after the first silence evidence. |

Scores between the thresholds retain the current evidence state. Policy values have
no library defaults.

### `WakePolicyConfig`

| Field | Constraint and effect |
| --- | --- |
| `score_threshold` | Finite probability in `[0, 1]`; a score at or above it is a hit. |
| `consecutive_score_count` | Positive number of consecutive hits required to activate. A miss resets the streak. |
| `silence_bridge_samples` | Nonnegative silence retained while completing a candidate. |
| `pre_roll_samples` | Nonnegative audio included before the first score in the confirming streak, clipped to candidate start. |

`pre_roll_samples` must cover the measured detection delay of the selected classifier
if the wake word must be included in activated audio.

### `StreamConfig`

Combines `vad_policy`, `wake_policy`, and positive
`max_retained_audio_samples`. For VAD frame size `V` and wake frame size `W`, the
constructor requires at least:

```text
pre_roll + consecutive_score_count * W + silence_bridge
  + max(left_padding + ceil(minimum_speech / V) * V,
        ceil(minimum_silence / V) * V)
```

The bound covers undecided steady-state PCM. It excludes the current copied caller
chunk and output arrays being assembled.

## `WakeStreamProcessor`

```python
WakeStreamProcessor(
    config,
    vad_backend,
    wake_backend,
    /,
    *,
    logger,
    diagnostic_event_capacity=0,
)
```

The logger is caller-configured and is bound internally with Wakelab context. A zero
diagnostic capacity disables event collection. A positive capacity creates a bounded
drop-old event queue.

| Member | Contract |
| --- | --- |
| `process(chunk)` | Open state only. Accepts the complete chunk and returns finalized output. |
| `rearm()` | Open state only. Clears wake evaluation and activation for future audio while preserving timeline and VAD state. |
| `discontinue()` | Open state only. Finalizes the current segment and forces a boundary on the next input. |
| `finish()` | Finalizes all real audio without model-frame padding and permanently ends input. |
| `drain_failed()` | Failed state only. Returns accepted audio conservatively without more inference. |
| `drain_diagnostics()` | Returns and clears diagnostics; available in every lifecycle state. |
| `close()` | Finalizes or drains, closes both backends, and becomes terminal. |
| `accepted_samples` | Processor-lifetime count accepted by `process()`. |
| `emitted_samples` | Processor-lifetime count returned from processor methods. |
| `retained_samples` | Current undecided PCM count. |

`StreamProcessor` is the structural `Protocol` for the seven methods above. It does
not include the concrete processor counters and is not runtime-checkable.

## Stream Errors

All stream errors derive from `StreamError`, which derives from `RuntimeError`.

| Error | Meaning |
| --- | --- |
| `StreamStateError` | Operation is invalid in the current lifecycle state. |
| `StreamProcessingError` | Processing/finalization failed after the input chunk may have been accepted. Use `drain_failed()`. |
| `StreamCloseError` | Finalization, drain, or backend closure failed. `outputs` contains recovered output and `errors` contains all underlying exceptions. |

`InputChunk` construction, configuration, and other checks before acceptance raise
ordinary `TypeError` or `ValueError`. Cross-chunk timing violations are detected
after the complete chunk is accepted and therefore raise `StreamProcessingError`;
do not retry that chunk, and recover it with `drain_failed()`.

## Backend Protocols

`lumivox_wakelab.vad.VadBackend` and
`lumivox_wakelab.wakeword.WakeWordBackend` use the same structural contract:

```python
class Backend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: NDArray[np.int16]) -> float: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

The sample rate must be 16,000 and frame size must be positive. `infer()` receives
exactly one native frame and returns a finite value in `[0, 1]`. It must be
transactional: an exception leaves stream state unchanged. `reset()` restores
deterministic pristine stream state. `close()` is terminal and idempotent.

## Concrete Backends

### `SileroVad`

Construct with `SileroVad.load(model_path, *, logger)`. Public read-only properties
are `sample_rate`, `frame_samples`, `model_version`, and `active_providers`. Methods
are `infer()`, `reset()`, and `close()`. The backend is also a context manager.

Silero errors are `VadError`, `VadInitializationError`, `VadInferenceError`, and
`VadClosedError`. `load()` may also propagate artifact missing or integrity errors if
the shared model changes after acquisition.

### `OpenWakeWord`

```python
OpenWakeWord.load(
    feature_model_dir,
    classifier_path,
    *,
    logger,
    classifier_bytes=None,
)
```

Supplying `classifier_bytes` loads that immutable snapshot while retaining
`classifier_path` as its diagnostic identity. Use this after the caller verifies a
classifier digest to avoid a read-after-verification race.

Public read-only properties are `sample_rate`, `frame_samples`, `feature_version`,
`classifier_history_frames`, and `active_providers`. Methods are `infer()`, `reset()`,
and `close()`. The backend is also a context manager.

Wake errors are `WakeWordError`, `WakeWordInitializationError`,
`WakeWordInferenceError`, and `WakeWordClosedError`. `load()` may also propagate
artifact missing or integrity errors if shared feature models change after
acquisition.

## Artifact Functions And Errors

```python
ensure_silero_vad_model(
    model_root, *, logger, allow_download=True, timeout=30.0
) -> pathlib.Path

ensure_openwakeword_feature_models(
    model_root, *, logger, allow_download=True, timeout=30.0
) -> pathlib.Path
```

Both verify actual artifact bytes against code-owned size and SHA-256 manifests.
Errors derive from `ArtifactError`: `ArtifactMissingError`,
`ArtifactIntegrityError`, `ArtifactDownloadError`, and
`ArtifactPublicationError`.

## Diagnostics

`drain_diagnostics()` returns `DiagnosticDrain(events, dropped_events)`. Events are
immutable and contain sample positions and scalar state, never PCM. Overflow drops
the oldest events and increments `dropped_events` until the next drain.

Public event types are:

- `VadFrameDiagnostic` and `VadSpeechDiagnostic`;
- `WakeCandidateStartedDiagnostic` and `WakeCandidateEndedDiagnostic`;
- `WakeFrameDiagnostic` and `ActivationDiagnostic`;
- `RearmDiagnostic` and `ContinuityBoundaryDiagnostic`;
- `FailureDiagnostic`.

Associated `StrEnum` types are `VadTransitionReason`, `WakeCandidateEndReason`,
`ContinuityBoundaryReason`, and `DiagnosticFailureOperation`. Event sample ranges are
processor-lifetime half-open ranges `[start_sample, end_sample)`.
