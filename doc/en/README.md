# Wakelab Documentation

Wakelab is the wake-word stage of a local voice pipeline. It synchronously
accepts continuous PCM audio and emits the same samples with independent
`is_speech` and `is_activated` metadata.

## Read This First

1. [Installation](installation.md) describes supported systems, Git installation,
   runtime dependencies, and model prerequisites.
2. [Library usage](usage.md) shows how to construct and operate a complete stream.
3. [API reference](api.md) defines public values, lifecycle methods, errors,
   diagnostics, and replaceable backend protocols.
4. [Models and artifacts](models.md) covers Silero VAD, openWakeWord, cache
   integrity, offline use, and classifier requirements.
5. [Demos and tools](demos.md) explains the microphone recorder, profile files,
   diagnostic workbench, and benchmark.
6. [Development](development.md) covers the repository workflow, architecture,
   tests, and contribution constraints.

For automated coding agents, use the standalone
[LLM integration guide](../llm/wakelab.md). A Russian translation starts at
[doc/ru/README.md](../ru/README.md).

## Scope

Wakelab owns synchronous stream processing, VAD policy, wake-word policy,
activation state, model adapters, and output metadata. The application owns:

- audio capture or file reading;
- conversion to the required audio format before calling Wakelab;
- wake-word classifier provenance, integrity policy, and licensing;
- speech-to-text, command recording, transport, and assistant state;
- serialization when several threads can access one processor;
- logging configuration and retry policy for setup failures.

Wakelab deliberately has no microphone API, resampler, multichannel API,
PyTorch runtime, asynchronous API, or internal model worker.

## Core Guarantees

- Every accepted sample is emitted exactly once and in input order.
- Samples are not resampled, synthesized, overlapped, or modified.
- Input may use arbitrary nonempty chunk lengths; model frame sizes stay private
  to the backends.
- Generation changes and discontinuities are hard model and timing boundaries.
- Activation remains latched until `rearm()`.
- Wake-word inference is bypassed while activation is latched.
- Downloads, session creation, validation, and warm-up occur outside `process()`.
- A terminal processing failure can be drained without additional inference.

The processor may buffer undecided audio, so an individual `process()` call can
return an empty list. Finite streams must be finalized to retrieve all accepted
audio.
