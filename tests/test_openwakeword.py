from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pytest

import lumivox_wakelab.wakeword as wakeword
from lumivox_wakelab.wakeword import (
    OpenWakeWord,
    WakeWordClosedError,
    WakeWordInferenceError,
    WakeWordInitializationError,
    _openwakeword,
    ensure_openwakeword_feature_models,
)
from lumivox_wakelab._artifacts import ArtifactManifest, ArtifactMissingError

from .test_artifacts import RecordingLogger


@dataclass
class Node:
    name: str
    type: str
    shape: list[int | str | None]


class FakeMelSession:
    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray[Any, Any]]] = []
        self.requested_outputs: list[list[str]] = []
        self.failure: Exception | None = None
        self.result_override: object | None = None

    def get_inputs(self) -> list[Node]:
        return [Node("input", "tensor(float)", ["batch_size", "samples"])]

    def get_outputs(self) -> list[Node]:
        return [Node("output", "tensor(float)", ["time", 1, "frames", 32])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> Any:
        self.requested_outputs.append(output_names)
        self.feeds.append(input_feed)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        if self.result_override is not None:
            return self.result_override
        row_count = input_feed["input"].shape[1] // 160 - 3
        rows = np.arange(row_count, dtype=np.float32).reshape(1, 1, row_count, 1)
        return [np.broadcast_to(rows, (1, 1, row_count, 32)).copy()]


class FakeEmbeddingSession:
    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray[Any, Any]]] = []
        self.requested_outputs: list[list[str]] = []
        self.failure: Exception | None = None
        self.result_override: object | None = None

    def get_inputs(self) -> list[Node]:
        return [Node("input_1", "tensor(float)", ["batch", 76, 32, 1])]

    def get_outputs(self) -> list[Node]:
        return [Node("conv2d_19", "tensor(float)", ["batch", 1, 1, 96])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> Any:
        self.requested_outputs.append(output_names)
        self.feeds.append(input_feed)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        if self.result_override is not None:
            return self.result_override
        windows = input_feed["input_1"]
        values = windows[:, -1, 0, 0].reshape(-1, 1, 1, 1)
        return [np.broadcast_to(values, (windows.shape[0], 1, 1, 96)).astype(np.float32).copy()]


class FakeClassifierSession:
    def __init__(self, history_frames: int = 28) -> None:
        self.history_frames = history_frames
        self.feeds: list[dict[str, np.ndarray[Any, Any]]] = []
        self.requested_outputs: list[list[str]] = []
        self.failure: Exception | None = None
        self.result_override: object | None = None

    def get_inputs(self) -> list[Node]:
        return [Node("custom_features", "tensor(float)", [1, self.history_frames, 96])]

    def get_outputs(self) -> list[Node]:
        return [Node("custom_score", "tensor(float)", [1, 1])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> Any:
        self.requested_outputs.append(output_names)
        self.feeds.append(input_feed)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        if self.result_override is not None:
            return self.result_override
        return [np.asarray([[0.25]], dtype=np.float32)]


def make_backend(
    mel: FakeMelSession | None = None,
    embedding: FakeEmbeddingSession | None = None,
    classifier: FakeClassifierSession | None = None,
) -> tuple[OpenWakeWord, FakeMelSession, FakeEmbeddingSession, FakeClassifierSession]:
    mel_session = mel or FakeMelSession()
    embedding_session = embedding or FakeEmbeddingSession()
    classifier_session = classifier or FakeClassifierSession()
    mel_template = np.zeros((76, 32), dtype=np.float32)
    embedding_template = np.broadcast_to(np.arange(120, dtype=np.float32).reshape(120, 1), (120, 96)).copy()
    backend = OpenWakeWord(
        mel_session,
        embedding_session,
        classifier_session,
        "custom_features",
        "custom_score",
        classifier_session.history_frames,
        mel_template,
        embedding_template,
        ("CPUExecutionProvider",),
        RecordingLogger(),
    )
    return backend, mel_session, embedding_session, classifier_session


def test_feature_manifests_and_acquisition_use_one_canonical_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    manifests = [_openwakeword.MELSPECTROGRAM_MANIFEST, _openwakeword.EMBEDDING_MANIFEST]
    assert [(item.filename, item.size, item.sha256) for item in manifests] == [
        (
            "melspectrogram.onnx",
            1_087_958,
            "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        ),
        (
            "embedding_model.onnx",
            1_326_578,
            "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        ),
    ]
    assert all(item.name == "openwakeword-features" and item.version == "v0.5.1" for item in manifests)
    assert all(item.url.endswith(f"/v0.5.1/{item.filename}") for item in manifests)

    calls: list[tuple[object, object, bool, float]] = []

    def fake_ensure(
        root: str | os.PathLike[str],
        manifest: ArtifactManifest,
        *,
        logger: object,
        allow_download: bool,
        timeout: float,
    ) -> Path:
        calls.append((root, manifest, allow_download, timeout))
        return Path(root) / "openwakeword-features" / "v0.5.1" / manifest.filename

    monkeypatch.setattr(wakeword, "ensure_artifact", fake_ensure)
    directory = ensure_openwakeword_feature_models(
        "models", logger=RecordingLogger(), allow_download=False, timeout=4.0
    )

    assert directory == Path("models/openwakeword-features/v0.5.1")
    assert [call[1] for call in calls] == manifests
    assert all(call[2:] == (False, 4.0) for call in calls)


def test_acquisition_stops_on_second_failure_but_keeps_first_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_ensure(*args: object, **kwargs: object) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArtifactMissingError("missing")
        return Path("models/openwakeword-features/v0.5.1/melspectrogram.onnx")

    monkeypatch.setattr(wakeword, "ensure_artifact", fake_ensure)
    with pytest.raises(ArtifactMissingError):
        ensure_openwakeword_feature_models("models", logger=RecordingLogger(), allow_download=False)
    assert calls == 2


def test_infer_uses_native_pipeline_without_pcm_normalization_and_preserves_input() -> None:
    backend, mel, embedding, classifier = make_backend()
    frame = np.arange(-640, 640, dtype=np.dtype("<i2"))
    original = frame.copy()

    assert backend.infer(frame) == 0.25

    assert np.array_equal(frame, original)
    assert mel.requested_outputs == [["output"]]
    assert mel.feeds[0]["input"].shape == (1, 1_760)
    assert mel.feeds[0]["input"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(mel.feeds[0]["input"][0, :480], np.zeros(480, dtype=np.float32))
    np.testing.assert_array_equal(mel.feeds[0]["input"][0, 480:], frame.astype(np.float32))

    mel_window = embedding.feeds[0]["input_1"]
    assert mel_window.shape == (1, 76, 32, 1)
    np.testing.assert_allclose(mel_window[0, -8:, 0, 0], np.arange(8, dtype=np.float32) / 10 + 2)
    assert classifier.requested_outputs == [["custom_score"]]
    features = classifier.feeds[0]["custom_features"]
    assert features.shape == (1, 28, 96)
    np.testing.assert_allclose(features[0, -1], np.full(96, 2.7, dtype=np.float32))


def test_raw_context_propagates_and_reset_restores_templates_without_inference() -> None:
    backend, mel, embedding, classifier = make_backend()
    first = np.arange(1_280, dtype=np.dtype("<i2"))
    second = np.zeros(1_280, dtype=np.dtype("<i2"))
    backend.infer(first)
    backend.infer(second)

    np.testing.assert_array_equal(mel.feeds[1]["input"][0, :480], first[-480:].astype(np.float32))
    assert classifier.feeds[1]["custom_features"][0, -2, 0] == pytest.approx(2.7)
    call_counts = (len(mel.feeds), len(embedding.feeds), len(classifier.feeds))

    backend.reset()
    assert (len(mel.feeds), len(embedding.feeds), len(classifier.feeds)) == call_counts
    backend.infer(second)
    np.testing.assert_array_equal(mel.feeds[2]["input"][0, :480], np.zeros(480, dtype=np.float32))
    assert classifier.feeds[2]["custom_features"][0, -2, 0] == 119.0


@pytest.mark.parametrize(
    ("frame", "error_type"),
    [
        ([0] * 1_280, TypeError),
        (np.zeros(0, dtype=np.dtype("<i2")), ValueError),
        (np.zeros((1, 1_280), dtype=np.dtype("<i2")), ValueError),
        (np.zeros(1_279, dtype=np.dtype("<i2")), ValueError),
        (np.zeros(1_281, dtype=np.dtype("<i2")), ValueError),
        (np.zeros(1_280, dtype=np.float32), TypeError),
        (np.zeros(1_280, dtype=np.dtype(">i2")), TypeError),
    ],
)
def test_invalid_frames_are_rejected(frame: object, error_type: type[Exception]) -> None:
    backend, _, _, _ = make_backend()
    with pytest.raises(error_type):
        backend.infer(frame)


@pytest.mark.parametrize("stage", ["mel", "embedding", "classifier"])
def test_stage_failure_does_not_commit_any_stream_state(stage: str) -> None:
    backend, mel, embedding, classifier = make_backend()
    first = np.full(1_280, 7, dtype=np.dtype("<i2"))
    if stage == "mel":
        mel.failure = RuntimeError("failed")
    elif stage == "embedding":
        embedding.failure = RuntimeError("failed")
    else:
        classifier.failure = RuntimeError("failed")

    with pytest.raises(WakeWordInferenceError):
        backend.infer(first)

    backend.infer(np.zeros(1_280, dtype=np.dtype("<i2")))
    np.testing.assert_array_equal(mel.feeds[-1]["input"][0, :480], np.zeros(480, dtype=np.float32))
    assert classifier.feeds[-1]["custom_features"][0, -2, 0] == 119.0


@pytest.mark.parametrize(
    ("stage", "result"),
    [
        ("mel", []),
        ("mel", [np.full((1, 1, 8, 32), np.nan, dtype=np.float32)]),
        ("embedding", [np.zeros((1, 96), dtype=np.float32)]),
        ("embedding", [np.full((1, 1, 1, 96), np.inf, dtype=np.float32)]),
        ("classifier", [np.asarray([[float("nan")]], dtype=np.float32)]),
        ("classifier", [np.asarray([[1.1]], dtype=np.float32)]),
    ],
)
def test_malformed_stage_results_are_rejected(stage: str, result: object) -> None:
    backend, mel, embedding, classifier = make_backend()
    if stage == "mel":
        mel.result_override = result
    elif stage == "embedding":
        embedding.result_override = result
    else:
        classifier.result_override = result
    with pytest.raises(WakeWordInferenceError):
        backend.infer(np.zeros(1_280, dtype=np.dtype("<i2")))


def test_close_is_idempotent_terminal_and_context_manager_closes() -> None:
    backend, _, _, _ = make_backend()
    backend.close()
    backend.close()
    with pytest.raises(WakeWordClosedError):
        backend.infer(np.zeros(1_280, dtype=np.dtype("<i2")))
    with pytest.raises(WakeWordClosedError):
        backend.reset()
    with pytest.raises(WakeWordClosedError):
        backend.__enter__()

    managed, _, _, _ = make_backend()
    with pytest.raises(RuntimeError), managed:
        raise RuntimeError("caller failure")
    with pytest.raises(WakeWordClosedError):
        managed.reset()


class FakeSessionOptions:
    inter_op_num_threads = 0
    intra_op_num_threads = 0
    graph_optimization_level: object = None


def fake_onnxruntime(
    mel: FakeMelSession,
    embedding: FakeEmbeddingSession,
    classifier: FakeClassifierSession,
    available: list[str] | None = None,
) -> SimpleNamespace:
    sessions = {b"mel": mel, b"embedding": embedding, b"classifier": classifier}
    captured: list[tuple[bytes, FakeSessionOptions, list[str]]] = []

    def inference_session(
        model: bytes, *, sess_options: FakeSessionOptions, providers: list[str]
    ) -> FakeMelSession | FakeEmbeddingSession | FakeClassifierSession:
        captured.append((model, sess_options, providers))
        session = sessions[model]
        assert isinstance(session, FakeMelSession | FakeEmbeddingSession | FakeClassifierSession)
        return session

    return SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: available or ["CPUExecutionProvider"],
        InferenceSession=inference_session,
        captured=captured,
    )


def configure_fake_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeMelSession, FakeEmbeddingSession, FakeClassifierSession, SimpleNamespace]:
    classifier_path = tmp_path / "classifier.onnx"
    classifier_path.write_bytes(b"classifier")
    mel = FakeMelSession()
    embedding = FakeEmbeddingSession()
    classifier = FakeClassifierSession()
    runtime = fake_onnxruntime(mel, embedding, classifier)

    def fake_read(path: object, manifest: object) -> bytes:
        if getattr(manifest, "filename") == "melspectrogram.onnx":
            return b"mel"
        return b"embedding"

    monkeypatch.setattr(_openwakeword, "read_verified_artifact", fake_read)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return classifier_path, mel, embedding, classifier, runtime


def test_load_constructs_three_byte_sessions_warms_pipeline_and_restores_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classifier_path, mel, embedding, classifier, runtime = configure_fake_load(tmp_path, monkeypatch)

    backend = OpenWakeWord.load("features", classifier_path, logger=RecordingLogger())

    assert [item[0] for item in runtime.captured] == [b"mel", b"embedding", b"classifier"]
    assert all(item[1] is runtime.captured[0][1] for item in runtime.captured)
    assert all(item[2] == ["CPUExecutionProvider"] for item in runtime.captured)
    options = runtime.captured[0][1]
    assert options.inter_op_num_threads == 1
    assert options.intra_op_num_threads == 1
    assert options.graph_optimization_level == "all"
    assert len(mel.feeds) == len(embedding.feeds) == 2
    assert len(classifier.feeds) == 1
    assert mel.feeds[0]["input"].shape == (1, 164_960)
    assert mel.feeds[1]["input"].shape == (1, 1_760)
    assert embedding.feeds[0]["input_1"].shape == (120, 76, 32, 1)
    assert embedding.feeds[1]["input_1"].shape == (1, 76, 32, 1)
    assert classifier.feeds[0]["custom_features"].shape == (1, 28, 96)

    assert backend.sample_rate == 16_000
    assert backend.frame_samples == 1_280
    assert backend.feature_version == "0.5.1"
    assert backend.classifier_history_frames == 28
    assert backend.active_providers == ("CPUExecutionProvider",)
    backend.infer(np.zeros(1_280, dtype=np.dtype("<i2")))
    np.testing.assert_array_equal(mel.feeds[2]["input"][0, :480], np.zeros(480, dtype=np.float32))
    backend.close()


def test_load_can_use_an_already_verified_classifier_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    classifier_path, _, _, _, runtime = configure_fake_load(tmp_path, monkeypatch)
    classifier_path.write_bytes(b"replaced")

    backend = OpenWakeWord.load(
        "features",
        classifier_path,
        logger=RecordingLogger(),
        classifier_bytes=b"classifier",
    )

    assert runtime.captured[-1][0] == b"classifier"
    backend.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda mel, embedding, classifier: setattr(
            mel, "get_outputs", lambda: [Node("wrong", "tensor(float)", ["time", 1, "frames", 32])]
        ),
        lambda mel, embedding, classifier: setattr(
            embedding, "get_inputs", lambda: [Node("input_1", "tensor(float)", ["batch", 75, 32, 1])]
        ),
        lambda mel, embedding, classifier: setattr(
            classifier, "get_inputs", lambda: [Node("features", "tensor(float)", [1, 121, 96])]
        ),
        lambda mel, embedding, classifier: setattr(
            classifier, "get_outputs", lambda: [Node("scores", "tensor(float)", [1, 2])]
        ),
    ],
)
def test_load_rejects_incompatible_feature_or_classifier_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
) -> None:
    classifier_path, mel, embedding, classifier, _ = configure_fake_load(tmp_path, monkeypatch)
    mutate(mel, embedding, classifier)
    with pytest.raises(WakeWordInitializationError):
        OpenWakeWord.load("features", classifier_path, logger=RecordingLogger())


def test_load_rejects_missing_classifier_and_provider_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mel, embedding, classifier, _ = configure_fake_load(tmp_path, monkeypatch)
    with pytest.raises(WakeWordInitializationError, match="Cannot read trusted"):
        OpenWakeWord.load("features", tmp_path / "missing.onnx", logger=RecordingLogger())

    classifier_path = tmp_path / "classifier.onnx"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        fake_onnxruntime(mel, embedding, classifier, ["AzureExecutionProvider"]),
    )
    with pytest.raises(WakeWordInitializationError, match="unavailable"):
        OpenWakeWord.load("features", classifier_path, logger=RecordingLogger())

    mel.get_providers = lambda: ["CPUExecutionProvider", "AzureExecutionProvider"]  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(mel, embedding, classifier))
    with pytest.raises(WakeWordInitializationError, match="unexpected providers"):
        OpenWakeWord.load("features", classifier_path, logger=RecordingLogger())
