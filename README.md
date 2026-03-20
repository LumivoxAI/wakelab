# Wakelab

Wakelab is a synchronous Python library for voice activity detection (VAD) and
wake-word activation in continuous audio streams. It accepts arbitrary-sized
chunks of 16 kHz mono signed 16-bit little-endian PCM, preserves every accepted
sample, and returns the audio annotated with speech and activation state.

The library provides:

- a stateful streaming processor with explicit lifecycle operations;
- Silero VAD and openWakeWord ONNX Runtime backends;
- explicit, integrity-checked acquisition of shared model artifacts;
- replaceable backend protocols for custom inference implementations;
- bounded opt-in diagnostics without retained PCM;
- a Linux microphone example and a local diagnostic workbench.

Wakelab does not capture microphone audio, run speech-to-text, resample audio,
or provide assistant orchestration. Wake-word classifiers are caller-supplied
trusted ONNX files and are not distributed by this project.

## Requirements

- Linux
- CPython 3.13 or 3.14
- mono, 16,000 Hz, PCM S16LE audio
- NumPy 2 and ONNX Runtime 1.24.1 or newer within the supported major version

Install the current `master` branch with `uv`:

```console
uv add "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

See the [installation guide](doc/en/installation.md) before using the library.

## Documentation

- [English documentation](doc/en/README.md)
- [Installation](doc/en/installation.md)
- [Library usage](doc/en/usage.md)
- [API reference](doc/en/api.md)
- [Models and artifacts](doc/en/models.md)
- [Demos and tools](doc/en/demos.md)
- [Development](doc/en/development.md)
- [Русская документация](doc/ru/README.md)
- [Standalone LLM integration guide](doc/llm/wakelab.md)

## Minimal Data Flow

```python
from lumivox_wakelab import InputChunk

outputs = processor.process(
    InputChunk(
        samples=captured.samples,
        running_time_ns=captured.running_time_ns,
        captured_at_ns=captured.captured_at_ns,
        generation=captured.generation,
        discontinuity=captured.discontinuity,
    )
)

for output in outputs:
    if output.is_activated:
        consume_activated_audio(output.samples)
```

`process()` may return zero or more chunks. Output boundaries do not match input
boundaries. Applications must also consume the chunks returned by `rearm()`,
`discontinue()`, `finish()`, `drain_failed()`, or `close()`, including
`StreamCloseError.outputs` when close fails.

## License

Wakelab is licensed under Apache-2.0. See [LICENSE](LICENSE). Model and upstream
notices are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
