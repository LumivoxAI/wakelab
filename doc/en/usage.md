# Library Usage

## Audio Contract

Each `InputChunk.samples` value must be a writable, nonempty, one-dimensional
`numpy.ndarray` with exact dtype `numpy.dtype("<i2")`. It represents mono PCM at
16,000 Hz. Wakelab does not accept bytes, floating-point audio, multiple channels,
or another sample rate.

Chunk lengths are arbitrary. Do not pre-frame input into the native model sizes.

## Construct A Processor

Model acquisition and backend loading are synchronous setup operations. The
following configuration values are copied from the repository's uncalibrated demo
profile; they are an example, not production defaults.

```python
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

# The application must configure lumivox-core logging and provide this value.
logger: Logger = application_logger
model_root = Path("model-data")
classifier_path = Path("models/my-wake-word.onnx")

vad_path = ensure_silero_vad_model(model_root, logger=logger)
feature_dir = ensure_openwakeword_feature_models(model_root, logger=logger)

vad = SileroVad.load(vad_path, logger=logger)
try:
    wake = OpenWakeWord.load(feature_dir, classifier_path, logger=logger)
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
    processor = WakeStreamProcessor(config, vad, wake, logger=logger)
except Exception:
    vad.close()
    wake.close()
    raise
```

Use `SileroVad.load()` and `OpenWakeWord.load()` for normal construction. Their
low-level constructors are not application entry points.

## Process Audio

```python
import numpy as np

from lumivox_wakelab import InputChunk, OutputChunk


def consume(outputs: list[OutputChunk]) -> None:
    for output in outputs:
        # Forward every output if downstream logic needs complete stream history.
        archive_pcm(output.samples)
        if output.is_activated:
            command_recorder.write(output.samples)


samples = np.asarray(source_samples, dtype=np.dtype("<i2")).copy()
consume(
    processor.process(
        InputChunk(
            samples=samples,
            running_time_ns=source_running_time_ns,
            captured_at_ns=source_wall_time_ns,
            generation=capture_generation,
            discontinuity=continuity_was_lost,
        )
    )
)
```

`process()` borrows caller memory only for the duration of the call and copies the
complete chunk before inference. The caller may reuse the array after the call
returns or raises. Every `OutputChunk.samples` array is independent, writable, and
owned by the receiver.

`process()` returns an eager list that may be empty. Output chunk sizes and
boundaries are implementation details. Each output has uniform `is_speech` and
`is_activated` values; activated silence is valid.

## Timestamps And Continuity

`running_time_ns` and `captured_at_ns` identify the first input sample. Output
timestamps are interpolated for the first output sample after rechunking:

```text
timestamp + floor(sample_offset * 1_000_000_000 / 16_000)
```

Use `running_time_ns=0` when stream-relative time is unavailable. An untimed
continuity segment remains untimed in all outputs. Do not mix zero and nonzero
running time in one segment. `captured_at_ns=0` is an ordinary wall-clock anchor,
not an unavailable-time sentinel.

Within a timed segment, input running-time anchors must agree with the sample
timeline within one sample period (62,500 ns). Ordering never depends on timestamps.

A changed `generation` or `discontinuity=True` starts a hard continuity boundary.
No VAD or wake-word model frame crosses it. Activation remains latched across the
boundary; call `rearm()` if the application also wants to clear activation.

Devicelab adaptation is a direct field mapping from `CapturedChunk` to `InputChunk`.
Devicelab callbacks are serialized, which matches the processor's threading contract.

## Lifecycle

Always consume outputs returned by lifecycle methods:

```python
consume(processor.rearm())       # clear activation, keep the current timeline
consume(processor.discontinue()) # finalize continuity, accept more input later
consume(processor.finish())      # finalize finite input and reject future input
consume(processor.close())       # finalize if needed and close both backends
```

Use `close()` in a `finally` block. `WakeStreamProcessor` is not a context manager.
`finish()` and `close()` are terminal for input and idempotent after successful
completion. An explicit discontinuity forces the next accepted input to carry a
discontinuity boundary without changing its generation.

No incomplete model frame is padded during a continuity boundary or finite-stream
finish. Real tail samples are still returned with conservative metadata.

## Processing Failures

If `process()`, `rearm()`, `discontinue()`, or finalization fails after audio was
accepted, the processor raises `StreamProcessingError` and enters a terminal failed
state. The failed call returns no output. Recover all accepted audio exactly once:

```python
from lumivox_wakelab import StreamCloseError, StreamProcessingError

try:
    consume(processor.process(chunk))
except StreamProcessingError:
    consume(processor.drain_failed())
    raise
finally:
    try:
        consume(processor.close())
    except StreamCloseError as error:
        consume(error.outputs)
        raise
```

`drain_failed()` performs no additional model inference. `close()` automatically
attempts the same drain if needed. If close itself raises `StreamCloseError`, consume
`error.outputs`; `error.errors` contains all resource exceptions.

## Threading

One processor and its two backends represent one logical stream. They are
synchronous, non-reentrant, and not thread-safe. Serialize all calls to `process()`,
lifecycle methods, counter properties, and `drain_diagnostics()`. Do not call one
processor concurrently from capture, shutdown, and UI threads. Use separate fully
initialized processor/backend instances for independent streams.

There are no internal workers or hidden asynchronous queues. `process()` runs model
inference in the caller's thread.
