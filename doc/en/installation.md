# Installation

## Supported Environment

Wakelab 0.1 supports CPython 3.13 and 3.14 on Linux. The public audio contract is
mono 16,000 Hz signed 16-bit little-endian PCM. ONNX execution is currently CPU
only.

The base package depends on:

- `lumivox-core` from its Git repository;
- `numpy>=2,<3`;
- `onnxruntime>=1.24.1,<2`.

Microphone capture through Devicelab and the NiceGUI workbench are repository
development dependencies, not runtime dependencies of the Wakelab package.

## Install From Git

The project is currently consumed from the `master` branch. With `uv`:

```console
uv add "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

With `pip`:

```console
python -m pip install "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

For reproducible deployments, replace `master` with a tested commit SHA in the
direct reference and commit the resulting lock file.

## Install A Development Checkout

The repository uses [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems/):

```console
git clone https://github.com/LumivoxAI/wakelab.git
cd wakelab
just postclone
```

`just postclone` creates the local environment and installs all dependency
groups. Use `just sync` after ordinary dependency changes.

## Model Prerequisites

The Python package contains no ONNX model files. Before constructing a processor:

1. Acquire the pinned Silero VAD model with `ensure_silero_vad_model()`.
2. Acquire the pinned openWakeWord feature models with
   `ensure_openwakeword_feature_models()`.
3. Provide a trusted local binary wake-word classifier compatible with the
   openWakeWord v0.5.1 feature ABI.

The first two operations can use the network and must run during setup, not in an
audio callback. See [Models and artifacts](models.md).

## Verify The Installation

```console
uv run python -c "import lumivox_wakelab; print('Wakelab import succeeded')"
```

Importing the package does not download or initialize models.
