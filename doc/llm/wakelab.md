# Wakelab Library Integration

Use this document as the complete integration contract for `lumivox-wakelab`.
Do not inspect or import Wakelab private modules. Do not depend on repository tools,
examples, profile helpers, or underscore-prefixed names.

## Purpose

Wakelab synchronously processes one continuous audio stream. It applies voice
activity detection and one wake-word classifier, then returns every accepted PCM
sample with two independent booleans:

- `is_speech`: VAD classification for the output range.
- `is_activated`: whether the wake activation latch applies to the output range.

Wakelab does not capture audio, resample, decode files, run STT, record commands,
configure application logging, or orchestrate an assistant.

## Supported Environment

- Linux.
- CPython 3.13 or 3.14.
- NumPy 2.x.
- ONNX Runtime 1.24.1 or newer, below 2.0.
- CPU execution only in the current implementation.
- Input audio is mono, 16,000 Hz, signed 16-bit little-endian PCM.

## Dependency

The package is currently installed from the `master` branch:

```console
uv add "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

Equivalent PEP 508 dependency:

```text
lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master
```

For a reproducible project, replace `master` with a tested Git commit SHA and commit
the dependency lock file.

The package pulls `lumivox-core`, NumPy, and ONNX Runtime. Do not add Silero VAD or
openWakeWord Python packages. Wakelab implements the required ONNX adapters directly.

## Required External Data

The installed package includes no model files.

Wakelab can acquire and verify these shared artifacts:

- Silero VAD v6.2.1 ONNX model.
- openWakeWord v0.5.1 mel-spectrogram and embedding ONNX models.

The application must provide one trusted local binary wake-word classifier ONNX
file. Wakelab never downloads a classifier. The classifier must have:

- exactly one float32 input shaped `[1, N, 96]`;
- `1 <= N <= 120`;
- exactly one float32 output shaped `[1, 1]`;
- compatibility with the openWakeWord v0.5.1 feature ABI;
- output interpreted as one finite score in `[0, 1]`.

The application owns classifier provenance, SHA-256 verification, license, quality,
threshold calibration, and temporal calibration. Do not load untrusted pickle files.

## Public Imports

Use only these modules:

```python
from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamConfig,
    VadPolicyConfig,
    WakePolicyConfig,
    StreamProcessor,
    WakeStreamProcessor,
    StreamError,
    StreamStateError,
    StreamProcessingError,
    StreamCloseError,
)

from lumivox_wakelab.vad import (
    VadBackend,
    SileroVad,
    ensure_silero_vad_model,
    ArtifactError,
    ArtifactMissingError,
    ArtifactIntegrityError,
    ArtifactDownloadError,
    ArtifactPublicationError,
    VadError,
    VadInitializationError,
    VadInferenceError,
    VadClosedError,
)

from lumivox_wakelab.wakeword import (
    WakeWordBackend,
    OpenWakeWord,
    ensure_openwakeword_feature_models,
    ArtifactError,
    ArtifactMissingError,
    ArtifactIntegrityError,
    ArtifactDownloadError,
    ArtifactPublicationError,
    WakeWordError,
    WakeWordInitializationError,
    WakeWordInferenceError,
    WakeWordClosedError,
)
```

Artifact error classes exported by `vad` and `wakeword` are the same classes. Import
them from one module only. Diagnostics exports are listed in the Diagnostics section.

## Setup Sequence

Perform this sequence before starting audio delivery:

1. Obtain a caller-configured `lumivox_core.logger.Logger`.
2. Read and verify the classifier into one immutable `bytes` snapshot.
3. Call both `ensure_*` functions. These may use the network.
4. Load `SileroVad` and `OpenWakeWord`. Loading creates ONNX sessions, validates
   them, and warms the complete inference paths.
5. Construct `WakeStreamProcessor` with explicit policy and retention values.
6. Start the audio source only after construction succeeds.

Do not call model acquisition or `load()` from an audio callback or from
`WakeStreamProcessor.process()`.

If the host application does not already configure `lumivox-core`, a standalone
development entry point can use this lifecycle:

```python
from lumivox_core.logger import (
    LogMode,
    LoggingConfig,
    configure_logging,
    get_logger,
    shutdown_logging,
)

configure_logging(
    LoggingConfig(application="my-application", mode=LogMode.DEV, level="INFO")
)
try:
    logger = get_logger(component="wake_pipeline")
    run_application(logger)
finally:
    shutdown_logging()
```

Configure logging once in the application entry point, not in reusable library code.

## Complete Construction Pattern

The policy values in this example are uncalibrated demo values. Do not describe them
as production defaults. Wakelab intentionally defines no policy defaults.

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from lumivox_core.logger import Logger
from lumivox_wakelab import (
    StreamConfig,
    VadPolicyConfig,
    WakePolicyConfig,
    WakeStreamProcessor,
)
from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model
from lumivox_wakelab.wakeword import (
    OpenWakeWord,
    ensure_openwakeword_feature_models,
)


def load_verified_bytes(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"classifier SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return data


def build_processor(
    *,
    logger: Logger,
    model_root: Path,
    classifier_path: Path,
    classifier_sha256: str,
    allow_download: bool,
) -> WakeStreamProcessor:
    classifier_bytes = load_verified_bytes(classifier_path, classifier_sha256)

    vad_path = ensure_silero_vad_model(
        model_root,
        logger=logger,
        allow_download=allow_download,
    )
    feature_dir = ensure_openwakeword_feature_models(
        model_root,
        logger=logger,
        allow_download=allow_download,
    )

    vad = SileroVad.load(vad_path, logger=logger)
    try:
        wake = OpenWakeWord.load(
            feature_dir,
            classifier_path,
            logger=logger,
            classifier_bytes=classifier_bytes,
        )
    except Exception:
        vad.close()
        raise

    config = StreamConfig(
        vad_policy=VadPolicyConfig(
            speech_threshold=0.5,
            silence_threshold=0.35,
            minimum_speech_samples=4_000,
            minimum_silence_samples=1_600,
            left_padding_samples=480,
            right_padding_samples=480,
        ),
        wake_policy=WakePolicyConfig(
            score_threshold=0.4,
            consecutive_score_count=3,
            silence_bridge_samples=8_000,
            pre_roll_samples=24_000,
        ),
        max_retained_audio_samples=160_000,
    )

    try:
        return WakeStreamProcessor(config, vad, wake, logger=logger)
    except Exception:
        vad.close()
        wake.close()
        raise
```

Use `classifier_bytes=` after digest verification. It guarantees that ONNX loads the
same classifier bytes that were verified. `classifier_path` remains required as the
diagnostic identity.

Use `SileroVad.load()` and `OpenWakeWord.load()`. Do not directly invoke their
low-level constructors.

## Artifact Acquisition

Signatures:

```python
ensure_silero_vad_model(
    model_root: str | os.PathLike[str],
    *,
    logger: Logger,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> pathlib.Path

ensure_openwakeword_feature_models(
    model_root: str | os.PathLike[str],
    *,
    logger: Logger,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> pathlib.Path
```

Cache locations:

```text
<model-root>/silero-vad/v6.2.1/silero_vad.onnx
<model-root>/openwakeword-features/v0.5.1/melspectrogram.onnx
<model-root>/openwakeword-features/v0.5.1/embedding_model.onnx
```

The functions verify actual bytes using code-owned size and SHA-256 manifests. A
`.sha256` sidecar is cached metadata, not a trust source. Download publication is
atomic. A bad or partial download does not replace a valid artifact.

Set `allow_download=False` after provisioning to prohibit network downloads. This
mode may still update sidecar metadata or lock files under `model_root`.

Artifact acquisition performs one bounded attempt. The calling application decides
whether and when to retry.

## Input Contract

Construct input as:

```python
InputChunk(
    samples: numpy.ndarray,
    running_time_ns: int,
    captured_at_ns: int,
    generation: int,
    discontinuity: bool,
)
```

Requirements for `samples`:

- actual `numpy.ndarray`, not bytes or an arbitrary array-like value;
- exact dtype `numpy.dtype("<i2")`;
- one-dimensional;
- nonempty;
- writable;
- contains mono PCM at exactly 16,000 Hz.

All integer metadata must be nonnegative Python integers. Booleans are rejected for
integer fields. `discontinuity` must be an actual `bool`.

Chunk sample count is arbitrary. Do not reshape input to 512 or 1,280 samples. The
processor performs model-specific framing internally.

`process()` copies the complete array before inference. The caller may mutate or
reuse the input array after `process()` returns or raises. Do not mutate it during
the call.

## Time Metadata

Both timestamps identify the first sample in the input chunk.

- `running_time_ns` is stream-relative time.
- `running_time_ns == 0` means unavailable for the entire continuity segment.
- Do not mix zero and nonzero `running_time_ns` within one segment.
- `captured_at_ns` is a wall-clock anchor.
- `captured_at_ns == 0` is not an unavailable sentinel. It is interpolated normally.

Output timestamps identify the first output sample and use:

```text
timestamp + floor(sample_offset * 1_000_000_000 / 16_000)
```

For a timed continuity segment, each new `running_time_ns` anchor must agree with
sample-count interpolation within 62,500 ns. Timing-mode or anchor violations are
detected after the complete input chunk is accepted. They raise
`StreamProcessingError`; do not retry that chunk, and use `drain_failed()`.

Output ordering is based on monotonic sample position, never on timestamps.

## Continuity Metadata

`generation` identifies a capture generation. Either condition starts a hard
continuity boundary:

- generation changes from the active segment;
- input has `discontinuity=True`.

At a hard boundary Wakelab finalizes pending real audio, resets model framing and
temporal state, and starts a new timing segment. No model frame combines samples from
opposite sides. Activation remains latched across the boundary. Call `rearm()`
separately if activation must be cleared.

The first output at an input discontinuity, generation change, or explicit
`discontinue()` boundary has `discontinuity=True`. Later output in that continuity
segment has `discontinuity=False`. Wakelab therefore synthesizes an output
discontinuity for a generation change even if the source flag was false.

For `lumivox_devicelab.CapturedChunk`, adapt fields directly:

```python
input_chunk = InputChunk(
    samples=captured.samples,
    running_time_ns=captured.running_time_ns,
    captured_at_ns=captured.captured_at_ns,
    generation=captured.generation,
    discontinuity=captured.discontinuity,
)
```

Do not make Wakelab depend on a Devicelab type at the processing boundary.

## Output Contract

`process()` returns `list[OutputChunk]`. The list may be empty because undecided
audio is retained. Each output contains:

```python
OutputChunk(
    samples,
    running_time_ns,
    captured_at_ns,
    generation,
    discontinuity,
    is_speech,
    is_activated,
)
```

Rules:

- Every accepted sample is eventually emitted exactly once and in original order.
- PCM values are unchanged.
- Output chunk boundaries do not correspond to input boundaries.
- Do not depend on a specific output chunk length.
- Metadata is uniform over one output chunk.
- `is_speech` and `is_activated` are independent.
- Activated silence is valid and expected after activation latches.
- Every output sample array is independent, writable, and receiver-owned.
- Outputs do not alias caller input, processor storage, or other outputs.

Consume outputs from every processor method, not only from `process()`.

## Policy Contract

Signatures:

```python
VadPolicyConfig(
    speech_threshold: float,
    silence_threshold: float,
    minimum_speech_samples: int,
    minimum_silence_samples: int,
    left_padding_samples: int,
    right_padding_samples: int,
)

WakePolicyConfig(
    score_threshold: float,
    consecutive_score_count: int,
    silence_bridge_samples: int,
    pre_roll_samples: int,
)

StreamConfig(
    vad_policy: VadPolicyConfig,
    wake_policy: WakePolicyConfig,
    max_retained_audio_samples: int,
)
```

Validation:

- probability fields are finite and in `[0, 1]`;
- `silence_threshold < speech_threshold`;
- minimum speech, minimum silence, and consecutive count are positive integers;
- left padding, right padding, silence bridge, and pre-roll are nonnegative integers;
- maximum retained audio is a positive integer and must satisfy the backend-specific
  retention formula.

For VAD frame size `V` and wake frame size `W`, minimum retention is:

```text
pre_roll_samples
+ consecutive_score_count * W
+ silence_bridge_samples
+ max(
    left_padding_samples + ceil(minimum_speech_samples / V) * V,
    ceil(minimum_silence_samples / V) * V,
  )
```

The concrete backends use `V = 512` and `W = 1280`. The processor validates the
actual injected backend sizes during construction.

Policy behavior:

- VAD score `>= speech_threshold` is speech evidence.
- VAD score `< silence_threshold` is silence evidence.
- A VAD score between thresholds preserves the prior evidence state.
- Wake score `>= score_threshold` is a hit.
- A lower wake score resets the consecutive-hit count.
- Armed non-speech does not invoke wake inference.
- A speech candidate can continue through no more than the configured silence bridge.
- Activation pre-roll starts before the first score in the confirming streak and is
  clipped to candidate start.
- Once activated, subsequent speech and silence remain activated until `rearm()`.
- Wake inference is bypassed while activated.

Calibrate all values against the selected classifier, acoustic environment, and
target hardware. `pre_roll_samples` must cover measured classifier detection delay to
include the complete wake word. Wakelab intentionally includes the wake word in
activated audio; it does not attempt exact wake-word removal.

## Processor API

Constructor:

```python
WakeStreamProcessor(
    config: StreamConfig,
    vad_backend: VadBackend,
    wake_backend: WakeWordBackend,
    /,
    *,
    logger: Logger,
    diagnostic_event_capacity: int = 0,
)
```

Methods:

```python
process(chunk: InputChunk, /) -> list[OutputChunk]
rearm() -> list[OutputChunk]
discontinue() -> list[OutputChunk]
finish() -> list[OutputChunk]
drain_failed() -> list[OutputChunk]
drain_diagnostics() -> DiagnosticDrain
close() -> list[OutputChunk]
```

Concrete processor counters:

```python
processor.accepted_samples  # lifetime accepted sample count
processor.emitted_samples   # lifetime returned sample count
processor.retained_samples  # current undecided sample count
```

Do not use `WakeStreamProcessor` as a context manager. It does not implement that
protocol. The concrete `SileroVad` and `OpenWakeWord` backends do implement context
managers, but normally ownership is transferred to `WakeStreamProcessor`, whose
`close()` closes both.

## Lifecycle

Use these operations exactly:

### `process(chunk)`

- Valid only in open state.
- Accepts the complete chunk before inference.
- Returns all output whose metadata became final.
- Can return an empty list.

### `rearm()`

- Valid only in open state.
- Cancels current wake evaluation.
- Clears latched activation at the boundary before future unaccepted audio.
- Resets wake backend state.
- Preserves VAD state, generation, and the continuous sample timeline.
- Returns any audio finalized by the operation.

Call this after the application finishes one activated command and wants to listen
for another wake word.

### `discontinue()`

- Valid only in open state.
- Finalizes the active continuity segment.
- Resets model and timing segment state.
- Forces the next accepted input to be marked discontinuous.
- Does not choose or increment the next input generation.
- Does not clear latched activation.
- Returns finalized audio.

Use this when continuity was lost outside an input chunk, for example after dropping
audio in an application queue.

### `finish()`

- Valid in open state.
- Finalizes all real pending audio.
- Does not synthesize padding for incomplete model frames.
- Permanently rejects future input.
- Returns finalized audio.
- Repeated successful calls return `[]`.

Use this for a finite source if resources do not need to be closed immediately.

### `close()`

- If open, finalizes audio as `finish()` would.
- If failed, attempts `drain_failed()`.
- Offers closure to both backends even if one fails.
- Is terminal and idempotent after the first call.
- Returns finalized or drained audio.

Always call it in a `finally` path and consume its output.

### `drain_failed()`

- Valid only after terminal processing failure.
- Returns withheld output and all remaining accepted audio exactly once.
- Performs no additional model inference.
- Uses conservative metadata for unresolved audio.
- Repeated successful calls return `[]`.

## Error Handling Pattern

A `StreamProcessingError` means the complete chunk passed to the failing `process()`
call was already accepted. Do not retry that chunk. Drain the failed processor and
construct a new processor if processing must continue.

```python
from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamCloseError,
    StreamProcessingError,
)


def handle_outputs(outputs: list[OutputChunk]) -> None:
    for output in outputs:
        downstream.accept(output)


errors: list[Exception] = []
try:
    for source_chunk in source:
        chunk = InputChunk(
            samples=source_chunk.samples,
            running_time_ns=source_chunk.running_time_ns,
            captured_at_ns=source_chunk.captured_at_ns,
            generation=source_chunk.generation,
            discontinuity=source_chunk.discontinuity,
        )
        try:
            handle_outputs(processor.process(chunk))
        except StreamProcessingError as error:
            handle_outputs(processor.drain_failed())
            errors.append(error)
            break
finally:
    try:
        handle_outputs(processor.close())
    except StreamCloseError as error:
        handle_outputs(error.outputs)
        errors.append(error)

if errors:
    raise ExceptionGroup("wake stream failed", errors)
```

Error hierarchy:

```text
StreamError
  StreamStateError
  StreamProcessingError
  StreamCloseError

ArtifactError
  ArtifactMissingError
  ArtifactIntegrityError
  ArtifactDownloadError
  ArtifactPublicationError

VadError
  VadInitializationError
  VadInferenceError
  VadClosedError

WakeWordError
  WakeWordInitializationError
  WakeWordInferenceError
  WakeWordClosedError
```

`StreamCloseError.outputs` is `list[OutputChunk]` recovered before/during closure.
`StreamCloseError.errors` is a tuple containing every underlying finalization,
drain, or resource-close exception. The processor is closed even when `close()`
raises.

`InputChunk` construction, configuration, and other validation before acceptance
raise `TypeError` or `ValueError`, not a stream error. Stateful cross-chunk timing
validation occurs after acceptance and is wrapped in `StreamProcessingError`.

Concrete backend `load()` methods can also propagate `ArtifactMissingError` or
`ArtifactIntegrityError` if a pinned shared artifact disappears or changes after its
acquisition step.

## Threading And Concurrency Guarantees

Treat one processor plus its VAD and wake backends as one single-thread-owned stream.

Guaranteed behavior when calls are serialized:

- synchronous eager method results;
- deterministic stream state for the same inputs and backend scores;
- no internal worker threads managed by Wakelab;
- no hidden asynchronous audio queue;
- no network or download in steady-state `process()`;
- model inference executes in the calling thread;
- backend inference state commits transactionally after successful inference;
- artifact acquisition coordinates cooperating Linux processes with a file lock and
  atomic publication.

Not guaranteed:

- thread safety of a processor or backend instance;
- runtime detection of concurrent or reentrant calls;
- safety of calling `close()` while another thread is in `process()`;
- parallel handling of two streams by one instance;
- an asyncio API;
- callback scheduling or deadline enforcement.

Required caller behavior:

- Serialize `process()`, `rearm()`, `discontinue()`, `finish()`, `drain_failed()`,
  `drain_diagnostics()`, `close()`, and counter reads for one instance.
- Do not hold an application lock in a way that allows reentrant calls from output
  handling or logging callbacks.
- Stop or detach the audio source before closing its processor.
- Create independent model backend and processor instances for independent streams.
- If an application transfers audio through a bounded queue and drops data, call
  `discontinue()` before processing later audio.

Devicelab serializes `CaptureHandler.on_chunk()` callbacks, so direct processing in
that callback satisfies the serialization requirement if no other thread calls the
same processor.

## Logging

Pass a caller-owned `lumivox_core.logger.Logger` to every setup/load/processor entry
point. Wakelab binds its own module/component context. The application, not Wakelab,
must configure and shut down logging.

Do not log PCM. Do not add application per-frame logging in the hot path. Use bounded
diagnostics for frame-level observability.

## Diagnostics

Diagnostics are disabled by default. Enable a bounded drop-old queue during
processor construction:

```python
processor = WakeStreamProcessor(
    config,
    vad,
    wake,
    logger=logger,
    diagnostic_event_capacity=4096,
)

drain = processor.drain_diagnostics()
for event in drain.events:
    diagnostic_sink.accept(event)
if drain.dropped_events:
    report_diagnostic_loss(drain.dropped_events)
```

`drain_diagnostics()` returns and clears events and the overflow count accumulated
since the preceding drain. It does not change stream processing state and remains
available after close. Disabled diagnostics return `DiagnosticDrain((), 0)`.

Diagnostics retain no PCM. They contain immutable sample positions, scores,
transitions, boundaries, and failures. Sample ranges are processor-lifetime half-open
ranges `[start_sample, end_sample)`.

Public diagnostic imports:

```python
from lumivox_wakelab import (
    DiagnosticEvent,
    DiagnosticDrain,
    VadFrameDiagnostic,
    VadSpeechDiagnostic,
    VadTransitionReason,
    WakeCandidateStartedDiagnostic,
    WakeCandidateEndedDiagnostic,
    WakeCandidateEndReason,
    WakeFrameDiagnostic,
    ActivationDiagnostic,
    RearmDiagnostic,
    ContinuityBoundaryDiagnostic,
    ContinuityBoundaryReason,
    FailureDiagnostic,
    DiagnosticFailureOperation,
)
```

Event fields:

```text
VadFrameDiagnostic:
  generation, start_sample, end_sample, probability

VadSpeechDiagnostic:
  generation, sample, is_speech, reason

WakeCandidateStartedDiagnostic:
  generation, start_sample

WakeCandidateEndedDiagnostic:
  generation, start_sample, end_sample, reason

WakeFrameDiagnostic:
  generation, start_sample, end_sample, score,
  threshold_met, consecutive_hit_count

ActivationDiagnostic:
  generation, activation_start_sample,
  first_start_sample, first_end_sample, first_score,
  confirming_start_sample, confirming_end_sample, confirming_score

RearmDiagnostic:
  generation, effective_sample

ContinuityBoundaryDiagnostic:
  sample, previous_generation, next_generation, reason

FailureDiagnostic:
  generation, retained_end_sample, accepted_end_sample,
  operation, error_type, message
```

Enum members and serialized string values:

```text
VadTransitionReason:
  EVIDENCE = "evidence"
  SEGMENT_END = "segment_end"
  PROCESSING_FAILURE = "processing_failure"

WakeCandidateEndReason:
  SILENCE_BRIDGE_EXCEEDED = "silence_bridge_exceeded"
  ACTIVATED = "activated"
  REARMED = "rearmed"
  CONTINUITY_BOUNDARY = "continuity_boundary"
  FINISHED = "finished"
  PROCESSING_FAILED = "processing_failed"

ContinuityBoundaryReason:
  INPUT_DISCONTINUITY = "input_discontinuity"
  GENERATION_CHANGE = "generation_change"
  GENERATION_CHANGE_AND_DISCONTINUITY = "generation_change_and_discontinuity"
  EXPLICIT_DISCONTINUE = "explicit_discontinue"

DiagnosticFailureOperation:
  PROCESS = "process"
  FINALIZE = "finalize"
  VAD_CLOSE = "vad_close"
  WAKE_CLOSE = "wake_close"
```

## Custom Backends

Use a custom backend only when the built-in model adapters are unsuitable. Implement
the appropriate structural protocol:

```python
from typing import Protocol
from numpy.typing import NDArray
import numpy as np


class Backend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: NDArray[np.int16]) -> float: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

Implementation requirements:

- `sample_rate` is exactly `16000`.
- `frame_samples` is a positive integer.
- One instance holds state for exactly one logical stream.
- `infer()` accepts exactly `frame_samples` PCM S16LE samples.
- `infer()` does not mutate its input.
- `infer()` returns one finite score in `[0, 1]`.
- If `infer()` raises, all backend stream state remains unchanged.
- `reset()` restores deterministic initial stream state.
- `close()` releases resources, is terminal, and is idempotent.
- The backend is synchronous, non-reentrant, and called serially.
- Initialization, model acquisition, schema validation, and warm-up finish before
  construction of `WakeStreamProcessor`.

The stream processor validates sample rate and frame size. It cannot validate a
custom wake classifier's detection delay. Configure `pre_roll_samples` from measured
behavior.

## Integration Checklist

- Install from Git and pin a tested commit for reproducibility.
- Run only on supported Linux and CPython versions.
- Supply mono 16 kHz writable NumPy PCM S16LE arrays.
- Verify the caller-owned classifier and pass the verified bytes to `OpenWakeWord.load()`.
- Acquire, load, validate, and warm every model before audio starts.
- Use explicit measured policy values and a sufficient retention bound.
- Map all five source metadata fields into `InputChunk` and preserve their documented
  output semantics.
- Consume zero or more outputs from every processing and lifecycle operation.
- Do not assume input and output chunk boundaries match.
- Serialize every operation on one processor instance.
- Mark every data loss as a discontinuity.
- Use `rearm()` only to clear wake activation for future audio.
- Use `finish()` for finite input and `close()` for final resource cleanup.
- On `StreamProcessingError`, do not retry the accepted chunk; call `drain_failed()`.
- On `StreamCloseError`, consume `error.outputs`.
- Do not import underscore-prefixed modules or repository tool/profile code.
