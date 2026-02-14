"""Stateful CPU ONNX adapter for the pinned openWakeWord feature ABI."""

from __future__ import annotations

import os
from typing import Any, Self, Protocol, Sequence, cast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from lumivox_core.logger import Logger

from lumivox_wakelab._artifacts import ArtifactError, ArtifactManifest, read_verified_artifact

_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 1_280
_RAW_CONTEXT_SAMPLES = 480
_MEL_ROWS_PER_FRAME = 8
_MEL_WINDOW_ROWS = 76
_MEL_BINS = 32
_EMBEDDING_SIZE = 96
_MAX_EMBEDDING_HISTORY = 120
_FEATURE_VERSION = "0.5.1"
_ARTIFACT_VERSION = f"v{_FEATURE_VERSION}"
_REQUESTED_PROVIDERS = ("CPUExecutionProvider",)
_FEATURE_MODEL_NAME = "openwakeword-features"
_MELSPECTROGRAM_FILENAME = "melspectrogram.onnx"
_EMBEDDING_FILENAME = "embedding_model.onnx"
_MEL_INPUT_NAME = "input"
_MEL_OUTPUT_NAME = "output"
_EMBEDDING_INPUT_NAME = "input_1"
_EMBEDDING_OUTPUT_NAME = "conv2d_19"

MELSPECTROGRAM_MANIFEST = ArtifactManifest(
    name=_FEATURE_MODEL_NAME,
    version=_ARTIFACT_VERSION,
    filename=_MELSPECTROGRAM_FILENAME,
    url=f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{_MELSPECTROGRAM_FILENAME}",
    size=1_087_958,
    sha256="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
)

EMBEDDING_MANIFEST = ArtifactManifest(
    name=_FEATURE_MODEL_NAME,
    version=_ARTIFACT_VERSION,
    filename=_EMBEDDING_FILENAME,
    url=f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{_EMBEDDING_FILENAME}",
    size=1_326_578,
    sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
)


class WakeWordError(RuntimeError):
    """Base class for wake-word backend failures."""


class WakeWordInitializationError(WakeWordError):
    """Raised when a wake-word backend cannot be initialized safely."""


class WakeWordInferenceError(WakeWordError):
    """Raised when an ONNX inference stage returns an invalid result."""


class WakeWordClosedError(WakeWordError):
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


def _is_dynamic(dimension: int | str | None) -> bool:
    return dimension is None or isinstance(dimension, str)


def _is_dimension(dimension: int | str | None, expected: int, *, dynamic: bool = True) -> bool:
    return dimension == expected or (dynamic and _is_dynamic(dimension))


def _single_node(nodes: Sequence[_NodeMetadata], kind: str, stage: str) -> _NodeMetadata:
    if len(nodes) != 1:
        raise WakeWordInitializationError(f"{stage} ONNX must have exactly one {kind}")
    return nodes[0]


def _validate_feature_schemas(mel_session: _InferenceSession, embedding_session: _InferenceSession) -> None:
    mel_input = _single_node(mel_session.get_inputs(), "input", "Mel-spectrogram")
    if (
        mel_input.name != _MEL_INPUT_NAME
        or mel_input.type != "tensor(float)"
        or len(mel_input.shape) != 2
        or not all(_is_dynamic(dimension) for dimension in mel_input.shape)
    ):
        raise WakeWordInitializationError(
            f"Mel-spectrogram ONNX input does not match the pinned contract: "
            f"{mel_input.name!r} {mel_input.type} {mel_input.shape}"
        )

    mel_output = _single_node(mel_session.get_outputs(), "output", "Mel-spectrogram")
    if (
        mel_output.name != _MEL_OUTPUT_NAME
        or mel_output.type != "tensor(float)"
        or len(mel_output.shape) != 4
        or not _is_dynamic(mel_output.shape[0])
        or not _is_dimension(mel_output.shape[1], 1)
        or not _is_dimension(mel_output.shape[2], 1)
        or not _is_dimension(mel_output.shape[3], _MEL_BINS, dynamic=False)
    ):
        raise WakeWordInitializationError(
            f"Mel-spectrogram ONNX output does not match the pinned contract: "
            f"{mel_output.name!r} {mel_output.type} {mel_output.shape}"
        )

    embedding_input = _single_node(embedding_session.get_inputs(), "input", "Embedding")
    if (
        embedding_input.name != _EMBEDDING_INPUT_NAME
        or embedding_input.type != "tensor(float)"
        or len(embedding_input.shape) != 4
        or not _is_dynamic(embedding_input.shape[0])
        or not _is_dimension(embedding_input.shape[1], _MEL_WINDOW_ROWS, dynamic=False)
        or not _is_dimension(embedding_input.shape[2], _MEL_BINS, dynamic=False)
        or not _is_dimension(embedding_input.shape[3], 1, dynamic=False)
    ):
        raise WakeWordInitializationError(
            f"Embedding ONNX input does not match the pinned contract: "
            f"{embedding_input.name!r} {embedding_input.type} {embedding_input.shape}"
        )

    embedding_output = _single_node(embedding_session.get_outputs(), "output", "Embedding")
    if (
        embedding_output.name != _EMBEDDING_OUTPUT_NAME
        or embedding_output.type != "tensor(float)"
        or len(embedding_output.shape) != 4
        or not _is_dynamic(embedding_output.shape[0])
        or not _is_dimension(embedding_output.shape[1], 1, dynamic=False)
        or not _is_dimension(embedding_output.shape[2], 1, dynamic=False)
        or not _is_dimension(embedding_output.shape[3], _EMBEDDING_SIZE, dynamic=False)
    ):
        raise WakeWordInitializationError(
            f"Embedding ONNX output does not match the pinned contract: "
            f"{embedding_output.name!r} {embedding_output.type} {embedding_output.shape}"
        )


def _validate_classifier_schema(session: _InferenceSession) -> tuple[str, str, int]:
    input_node = _single_node(session.get_inputs(), "input", "Wake-word classifier")
    output_node = _single_node(session.get_outputs(), "output", "Wake-word classifier")

    shape = input_node.shape
    if (
        input_node.type != "tensor(float)"
        or len(shape) != 3
        or shape[0] != 1
        or not isinstance(shape[1], int)
        or isinstance(shape[1], bool)
        or shape[1] <= 0
        or shape[1] > _MAX_EMBEDDING_HISTORY
        or shape[2] != _EMBEDDING_SIZE
    ):
        raise WakeWordInitializationError(
            f"Wake-word classifier ONNX input must be float32 [1, N, 96] with "
            f"1 <= N <= {_MAX_EMBEDDING_HISTORY}: {input_node.type} {shape}"
        )
    if output_node.type != "tensor(float)" or output_node.shape != [1, 1]:
        raise WakeWordInitializationError(
            f"Wake-word classifier ONNX output must be float32 [1, 1]: {output_node.type} {output_node.shape}"
        )
    return input_node.name, output_node.name, shape[1]


def _run_mel(session: _InferenceSession, audio: NDArray[np.float32], expected_rows: int) -> NDArray[np.float32]:
    try:
        results = session.run([_MEL_OUTPUT_NAME], {_MEL_INPUT_NAME: audio})
    except Exception as exc:
        raise WakeWordInferenceError("Mel-spectrogram ONNX inference failed") from exc
    if not isinstance(results, list | tuple) or len(results) != 1:
        raise WakeWordInferenceError("Mel-spectrogram ONNX returned an invalid result count")
    output = results[0]
    expected_shape = (1, 1, expected_rows, _MEL_BINS)
    if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != expected_shape:
        raise WakeWordInferenceError(
            f"Mel-spectrogram ONNX returned an invalid tensor; expected {expected_shape} float32"
        )
    transformed = output[0, 0, :, :] / np.float32(10.0) + np.float32(2.0)
    if not np.isfinite(transformed).all():
        raise WakeWordInferenceError("Mel-spectrogram ONNX returned non-finite values")
    return transformed


def _run_embedding(
    session: _InferenceSession, windows: NDArray[np.float32], expected_count: int
) -> NDArray[np.float32]:
    try:
        results = session.run([_EMBEDDING_OUTPUT_NAME], {_EMBEDDING_INPUT_NAME: windows})
    except Exception as exc:
        raise WakeWordInferenceError("Embedding ONNX inference failed") from exc
    if not isinstance(results, list | tuple) or len(results) != 1:
        raise WakeWordInferenceError("Embedding ONNX returned an invalid result count")
    output = results[0]
    expected_shape = (expected_count, 1, 1, _EMBEDDING_SIZE)
    if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != expected_shape:
        raise WakeWordInferenceError(f"Embedding ONNX returned an invalid tensor; expected {expected_shape} float32")
    embeddings = output[:, 0, 0, :]
    if not np.isfinite(embeddings).all():
        raise WakeWordInferenceError("Embedding ONNX returned non-finite values")
    return embeddings


def _run_classifier(
    session: _InferenceSession,
    input_name: str,
    output_name: str,
    features: NDArray[np.float32],
) -> float:
    try:
        results = session.run([output_name], {input_name: features})
    except Exception as exc:
        raise WakeWordInferenceError("Wake-word classifier ONNX inference failed") from exc
    if not isinstance(results, list | tuple) or len(results) != 1:
        raise WakeWordInferenceError("Wake-word classifier ONNX returned an invalid result count")
    output = results[0]
    if not isinstance(output, np.ndarray) or output.dtype != np.dtype(np.float32) or output.shape != (1, 1):
        raise WakeWordInferenceError("Wake-word classifier ONNX returned an invalid score tensor")
    score = float(output[0, 0])
    if not np.isfinite(score):
        raise WakeWordInferenceError("Wake-word classifier ONNX returned a non-finite score")
    if score < 0.0 or score > 1.0:
        raise WakeWordInferenceError(f"Wake-word classifier ONNX returned an out-of-range score: {score}")
    return score


def _silence_templates(
    mel_session: _InferenceSession, embedding_session: _InferenceSession
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    required_mel_rows = _MEL_WINDOW_ROWS + _MEL_ROWS_PER_FRAME * (_MAX_EMBEDDING_HISTORY - 1)
    silence_samples = (required_mel_rows + 3) * 160
    silence = np.zeros((1, silence_samples), dtype=np.float32)
    mel = _run_mel(mel_session, silence, required_mel_rows)
    windows = np.stack([mel[offset : offset + _MEL_WINDOW_ROWS] for offset in range(0, required_mel_rows - 75, 8)])[
        ..., None
    ].astype(np.float32, copy=False)
    if windows.shape != (_MAX_EMBEDDING_HISTORY, _MEL_WINDOW_ROWS, _MEL_BINS, 1):
        raise WakeWordInitializationError("Failed to construct deterministic openWakeWord silence windows")
    embeddings = _run_embedding(embedding_session, windows, _MAX_EMBEDDING_HISTORY)
    return mel[-_MEL_WINDOW_ROWS:].copy(), embeddings.copy()


class OpenWakeWord:
    """One-stream, stateful openWakeWord binary classifier backend."""

    def __init__(
        self,
        mel_session: _InferenceSession,
        embedding_session: _InferenceSession,
        classifier_session: _InferenceSession,
        classifier_input_name: str,
        classifier_output_name: str,
        classifier_history_frames: int,
        mel_template: NDArray[np.float32],
        embedding_template: NDArray[np.float32],
        providers: tuple[str, ...],
        logger: Logger,
    ) -> None:
        self._mel_session: _InferenceSession | None = mel_session
        self._embedding_session: _InferenceSession | None = embedding_session
        self._classifier_session: _InferenceSession | None = classifier_session
        self._classifier_input_name = classifier_input_name
        self._classifier_output_name = classifier_output_name
        self._classifier_history_frames = classifier_history_frames
        self._mel_template: NDArray[np.float32] | None = mel_template
        self._embedding_template: NDArray[np.float32] | None = embedding_template
        self._active_providers = providers
        self._logger = logger
        self._raw_context: NDArray[np.float32] | None = None
        self._mel_history: NDArray[np.float32] | None = None
        self._embedding_history: NDArray[np.float32] | None = None
        self._restore_stream_state()

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return _FRAME_SAMPLES

    @property
    def feature_version(self) -> str:
        return _FEATURE_VERSION

    @property
    def classifier_history_frames(self) -> int:
        return self._classifier_history_frames

    @property
    def active_providers(self) -> tuple[str, ...]:
        return self._active_providers

    @classmethod
    def load(
        cls,
        feature_model_dir: str | os.PathLike[str],
        classifier_path: str | os.PathLike[str],
        *,
        logger: Logger,
    ) -> Self:
        """Return a verified, warmed, deterministic openWakeWord backend."""

        bound_logger = logger.bind(module="wakelab", component="openwakeword")
        feature_directory = Path(feature_model_dir)
        classifier_path_object = Path(classifier_path)
        mel_session: _InferenceSession | None = None
        embedding_session: _InferenceSession | None = None
        classifier_session: _InferenceSession | None = None
        instance: Self | None = None
        try:
            mel_bytes = read_verified_artifact(feature_directory / _MELSPECTROGRAM_FILENAME, MELSPECTROGRAM_MANIFEST)
            embedding_bytes = read_verified_artifact(feature_directory / _EMBEDDING_FILENAME, EMBEDDING_MANIFEST)
            try:
                classifier_bytes = classifier_path_object.read_bytes()
            except OSError as exc:
                raise WakeWordInitializationError(
                    f"Cannot read trusted wake-word classifier: {classifier_path_object}"
                ) from exc

            import onnxruntime as ort  # type: ignore[import-untyped]

            available_providers = tuple(ort.get_available_providers())
            if _REQUESTED_PROVIDERS[0] not in available_providers:
                raise WakeWordInitializationError(
                    f"CPUExecutionProvider is unavailable; available providers: {available_providers}"
                )
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sessions: list[_InferenceSession] = []
            for model_bytes in (mel_bytes, embedding_bytes, classifier_bytes):
                session = cast(
                    _InferenceSession,
                    ort.InferenceSession(model_bytes, sess_options=options, providers=list(_REQUESTED_PROVIDERS)),
                )
                providers = tuple(session.get_providers())
                if providers != _REQUESTED_PROVIDERS:
                    raise WakeWordInitializationError(
                        f"openWakeWord activated unexpected providers: {providers}; expected {_REQUESTED_PROVIDERS}"
                    )
                sessions.append(session)
            mel_session, embedding_session, classifier_session = sessions

            _validate_feature_schemas(mel_session, embedding_session)
            classifier_input, classifier_output, history_frames = _validate_classifier_schema(classifier_session)
            mel_template, embedding_template = _silence_templates(mel_session, embedding_session)
            classifier_features = embedding_template[-history_frames:][None, :, :].astype(np.float32, copy=False)
            _run_classifier(classifier_session, classifier_input, classifier_output, classifier_features)

            instance = cls(
                mel_session,
                embedding_session,
                classifier_session,
                classifier_input,
                classifier_output,
                history_frames,
                mel_template,
                embedding_template,
                _REQUESTED_PROVIDERS,
                bound_logger,
            )
        except ArtifactError as exc:
            bound_logger.error("openwakeword_initialization_failed", error=str(exc))
            raise
        except WakeWordInitializationError as exc:
            if instance is not None:
                instance.close()
            mel_session = embedding_session = classifier_session = None
            bound_logger.error("openwakeword_initialization_failed", error=str(exc))
            raise WakeWordInitializationError(str(exc)) from None
        except Exception as exc:
            if instance is not None:
                instance.close()
            mel_session = embedding_session = classifier_session = None
            bound_logger.error("openwakeword_initialization_failed", error=str(exc))
            raise WakeWordInitializationError(
                f"Failed to initialize openWakeWord classifier from {classifier_path_object}"
            ) from exc

        bound_logger.info(
            "openwakeword_initialized",
            providers=_REQUESTED_PROVIDERS,
            classifier_history_frames=history_frames,
        )
        return instance

    def _restore_stream_state(self) -> None:
        assert self._mel_template is not None
        assert self._embedding_template is not None
        self._raw_context = np.zeros((1, _RAW_CONTEXT_SAMPLES), dtype=np.float32)
        self._mel_history = self._mel_template.copy()
        self._embedding_history = self._embedding_template.copy()

    def _require_open(
        self,
    ) -> tuple[
        _InferenceSession,
        _InferenceSession,
        _InferenceSession,
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        if self._mel_session is None or self._embedding_session is None or self._classifier_session is None:
            raise WakeWordClosedError("openWakeWord backend is closed")
        assert self._raw_context is not None
        assert self._mel_history is not None
        assert self._embedding_history is not None
        return (
            self._mel_session,
            self._embedding_session,
            self._classifier_session,
            self._raw_context,
            self._mel_history,
            self._embedding_history,
        )

    def infer(self, frame: NDArray[np.int16]) -> float:
        """Return one raw score for one native 1,280-sample PCM16 frame."""

        mel_session, embedding_session, classifier_session, raw_context, mel_history, embedding_history = (
            self._require_open()
        )
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

        float_frame = frame.astype(np.float32).reshape(1, _FRAME_SAMPLES)
        mel_input = np.concatenate((raw_context, float_frame), axis=1)
        new_mel = _run_mel(mel_session, mel_input, _MEL_ROWS_PER_FRAME)
        candidate_mel = np.concatenate((mel_history, new_mel), axis=0)[-_MEL_WINDOW_ROWS:]
        embedding_input = candidate_mel[None, :, :, None].astype(np.float32, copy=False)
        new_embedding = _run_embedding(embedding_session, embedding_input, 1)
        candidate_embeddings = np.concatenate((embedding_history, new_embedding), axis=0)[-_MAX_EMBEDDING_HISTORY:]
        classifier_input = candidate_embeddings[-self._classifier_history_frames :][None, :, :].astype(
            np.float32, copy=False
        )
        score = _run_classifier(
            classifier_session,
            self._classifier_input_name,
            self._classifier_output_name,
            classifier_input,
        )

        self._raw_context = float_frame[:, -_RAW_CONTEXT_SAMPLES:].copy()
        self._mel_history = candidate_mel.copy()
        self._embedding_history = candidate_embeddings.copy()
        return score

    def reset(self) -> None:
        """Restore deterministic stream history without invoking ONNX."""

        self._require_open()
        self._restore_stream_state()
        self._logger.debug("openwakeword_reset")

    def close(self) -> None:
        """Release all sessions and stream/template arrays. Closure is terminal."""

        if self._mel_session is None:
            return
        self._mel_session = None
        self._embedding_session = None
        self._classifier_session = None
        self._raw_context = None
        self._mel_history = None
        self._embedding_history = None
        self._mel_template = None
        self._embedding_template = None
        self._logger.info("openwakeword_closed")

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
