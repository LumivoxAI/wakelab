"""Stateful CPU ONNX adapter for the pinned Silero VAD model."""

from __future__ import annotations

import os
from typing import Any, Self, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from lumivox_core.logger import Logger

from lumivox_wakelab._artifacts import ArtifactError, ArtifactManifest, read_verified_artifact

_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 512
_MODEL_VERSION = "6.2.1"
_ARTIFACT_VERSION = f"v{_MODEL_VERSION}"
_CONTEXT_SAMPLES = 64
_MODEL_INPUT_SAMPLES = _FRAME_SAMPLES + _CONTEXT_SAMPLES
_STATE_SHAPE = (2, 1, 128)
_REQUESTED_PROVIDERS = ("CPUExecutionProvider",)

SILERO_VAD_MANIFEST = ArtifactManifest(
    name="silero-vad",
    version=_ARTIFACT_VERSION,
    filename="silero_vad.onnx",
    url=("https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx"),
    size=2_327_524,
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)


class VadError(RuntimeError):
    """Base class for VAD backend failures."""


class VadInitializationError(VadError):
    """Raised when a VAD backend cannot be initialized safely."""


class VadInferenceError(VadError):
    """Raised when ONNX inference returns an invalid result."""


class VadClosedError(VadError):
    """Raised when a terminally closed backend is used."""


class _NodeMetadata(Protocol):
    name: str
    type: str
    shape: list[int | str | None]


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_NodeMetadata]: ...

    def get_outputs(self) -> Sequence[_NodeMetadata]: ...

    def get_providers(self) -> list[str]: ...

    def run(self, output_names: list[str], input_feed: dict[str, NDArray[Any]]) -> list[Any]: ...


def _shape_is_compatible(actual: Sequence[int | str | None], expected: tuple[int, ...]) -> bool:
    return len(actual) == len(expected) and all(
        dimension == expected_dimension or dimension is None or isinstance(dimension, str)
        for dimension, expected_dimension in zip(actual, expected, strict=True)
    )


def _validate_nodes(
    nodes: Sequence[_NodeMetadata],
    expected: dict[str, tuple[str, tuple[int, ...]]],
    kind: str,
) -> None:
    by_name = {node.name: node for node in nodes}
    if len(by_name) != len(nodes) or set(by_name) != set(expected):
        raise VadInitializationError(f"Silero ONNX {kind} names do not match the pinned contract: {sorted(by_name)}")
    for name, (expected_type, expected_shape) in expected.items():
        node = by_name[name]
        if node.type != expected_type or not _shape_is_compatible(node.shape, expected_shape):
            raise VadInitializationError(
                f"Silero ONNX {kind} {name!r} has incompatible type or shape: {node.type} {node.shape}"
            )


def _validate_schema(session: _InferenceSession) -> None:
    _validate_nodes(
        session.get_inputs(),
        {
            "input": ("tensor(float)", (1, _MODEL_INPUT_SAMPLES)),
            "state": ("tensor(float)", _STATE_SHAPE),
            "sr": ("tensor(int64)", ()),
        },
        "input",
    )
    _validate_nodes(
        session.get_outputs(),
        {
            "output": ("tensor(float)", (1, 1)),
            "stateN": ("tensor(float)", _STATE_SHAPE),
        },
        "output",
    )


class SileroVad:
    """One-stream, stateful Silero VAD probability backend."""

    def __init__(self, session: _InferenceSession, providers: tuple[str, ...], logger: Logger) -> None:
        self._session: _InferenceSession | None = session
        self._active_providers = providers
        self._logger = logger
        self._state: NDArray[np.float32] | None = None
        self._context: NDArray[np.float32] | None = None
        self._sample_rate_input: NDArray[np.int64] | None = None
        self._initialize_stream_state()

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return _FRAME_SAMPLES

    @property
    def model_version(self) -> str:
        return _MODEL_VERSION

    @property
    def active_providers(self) -> tuple[str, ...]:
        return self._active_providers

    @classmethod
    def load(cls, model_path: str | os.PathLike[str], *, logger: Logger) -> Self:
        """Return a verified, initialized, warmed, and pristine backend.

        This synchronous setup operation never accesses the network. It loads
        the model bytes, creates and validates the ONNX session, runs one warm-up
        inference, and resets all stream state before returning.
        """

        bound_logger = logger.bind(module="wakelab", component="silero_vad")
        path_string = os.fspath(model_path)
        try:
            model_bytes = read_verified_artifact(model_path, SILERO_VAD_MANIFEST)
        except ArtifactError as exc:
            bound_logger.error("silero_vad_initialization_failed", path=path_string, error=str(exc))
            raise
        bound_logger.info("silero_vad_initializing", path=path_string)

        session: _InferenceSession | None = None
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            available_providers = tuple(ort.get_available_providers())
            if _REQUESTED_PROVIDERS[0] not in available_providers:
                raise VadInitializationError(
                    f"CPUExecutionProvider is unavailable; available providers: {available_providers}"
                )

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = cast(
                _InferenceSession,
                ort.InferenceSession(
                    model_bytes,
                    sess_options=options,
                    providers=list(_REQUESTED_PROVIDERS),
                ),
            )
            providers = tuple(session.get_providers())
            if providers != _REQUESTED_PROVIDERS:
                raise VadInitializationError(
                    f"Silero VAD activated unexpected providers: {providers}; expected {_REQUESTED_PROVIDERS}"
                )
            _validate_schema(session)
            instance = cls(session, providers, bound_logger)
            try:
                instance.infer(np.zeros(_FRAME_SAMPLES, dtype=np.dtype("<i2")))
                instance.reset()
            except Exception:
                instance.close()
                raise
        except VadInitializationError as exc:
            session = None
            message = str(exc)
            bound_logger.error("silero_vad_initialization_failed", path=path_string, error=message)
            raise VadInitializationError(message) from None
        except Exception as exc:
            session = None
            bound_logger.error("silero_vad_initialization_failed", path=path_string, error=str(exc))
            raise VadInitializationError(f"Failed to initialize Silero VAD from {path_string}") from None

        bound_logger.info("silero_vad_initialized", providers=providers)
        return instance

    def _initialize_stream_state(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)
        self._sample_rate_input = np.asarray(_SAMPLE_RATE, dtype=np.int64)

    def _require_open(self) -> tuple[_InferenceSession, NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        if self._session is None:
            raise VadClosedError("Silero VAD is closed")
        assert self._state is not None
        assert self._context is not None
        assert self._sample_rate_input is not None
        return self._session, self._state, self._context, self._sample_rate_input

    def infer(self, frame: NDArray[np.int16]) -> float:
        """Return a speech probability for one native 512-sample PCM16 frame."""

        session, state, context, sample_rate_input = self._require_open()
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 1:
            raise ValueError("frame must be one-dimensional")
        if frame.size == 0:
            raise ValueError("frame must not be empty")
        if frame.dtype != np.dtype("<i2"):
            raise TypeError('frame dtype must be numpy.dtype("<i2")')
        if frame.size != _FRAME_SAMPLES:
            raise ValueError(f"frame must contain exactly {_FRAME_SAMPLES} samples")

        normalized = frame.astype(np.float32).reshape(1, _FRAME_SAMPLES) / np.float32(32768.0)
        model_input = np.concatenate((context, normalized), axis=1)
        try:
            results = session.run(
                ["output", "stateN"],
                {"input": model_input, "state": state, "sr": sample_rate_input},
            )
        except Exception as exc:
            raise VadInferenceError("Silero ONNX inference failed") from exc

        if not isinstance(results, list | tuple) or len(results) != 2:
            raise VadInferenceError("Silero ONNX returned an invalid result count")
        output, next_state = results
        if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != (1, 1):
            raise VadInferenceError("Silero ONNX returned an invalid probability tensor")
        if (
            not isinstance(next_state, np.ndarray)
            or next_state.dtype != np.dtype(np.float32)
            or next_state.shape != _STATE_SHAPE
        ):
            raise VadInferenceError("Silero ONNX returned an invalid recurrent state tensor")
        if not np.isfinite(output).all() or not np.isfinite(next_state).all():
            raise VadInferenceError("Silero ONNX returned non-finite values")

        probability = float(output[0, 0])
        if probability < 0.0 or probability > 1.0:
            raise VadInferenceError(f"Silero ONNX returned an out-of-range probability: {probability}")

        self._state = next_state
        self._context = normalized[:, -_CONTEXT_SAMPLES:].copy()
        return probability

    def reset(self) -> None:
        """Clear stream-dependent recurrent state while preserving the session."""

        self._require_open()
        self._initialize_stream_state()
        self._logger.debug("silero_vad_reset")

    def close(self) -> None:
        """Release resources owned by this wrapper. Closure is terminal."""

        if self._session is None:
            return
        self._session = None
        self._state = None
        self._context = None
        self._sample_rate_input = None
        self._logger.info("silero_vad_closed")

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
