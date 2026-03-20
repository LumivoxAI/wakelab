# Demos And Tools

The commands on this page run from a development checkout prepared with
`just postclone`. They are repository tools, not installed package APIs.

## Microphone Wake Recorder

The Linux/PipeWire example uses Devicelab to listen for a wake word and writes a
bounded activated region as mono 16 kHz PCM16 WAV.

List devices without loading ONNX models:

```console
uv run python -m examples.microphone_wake_recorder --list-microphones
```

Use the default device and record ten seconds of activated audio:

```console
uv run python -m examples.microphone_wake_recorder
```

Select explicit values:

```console
uv run python -m examples.microphone_wake_recorder \
  --microphone-id 'device-id-from-list' \
  --profile examples/profile.json \
  --output recordings/command.wav \
  --duration 5
```

The destination directory must exist and the destination file must not exist.
Ctrl+C writes already collected activated audio as a partial recording if any exists.

The default `examples/profile.json` expects a private local classifier at the path
and digest recorded in that profile. That classifier is not part of the repository.
Copy the profile and set your classifier path, SHA-256, policy, and retention values.
Relative paths are resolved from the profile file, not the current shell directory.

## Diagnostic Workbench

Start the local NiceGUI workbench:

```console
uv run python tools/wake_pipeline_lab.py --profile tools/profile.json
```

The server binds only to `127.0.0.1`. It displays bounded input, VAD, wake score,
state transition, and diagnostic queue history. Profile edits made while listening
take effect on the next Start. Every Start builds and warms fresh backends.

The workbench is intended for interactive diagnosis and classifier-specific policy
calibration. Its profile is not a production default and can intentionally differ
from the microphone example profile.

## Streaming Benchmark

Measure the synthetic retained-audio path:

```console
uv run python tools/benchmark_streaming.py
```

Pass `--feature-dir`, `--classifier`, and `--openwakeword-frames` to measure real
openWakeWord steady-state inference after warm-up. Use
`uv run python tools/benchmark_streaming.py --help` for all options.

Benchmark results are hardware-specific observations, not latency guarantees.
