from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from dataclasses import dataclass

import numpy as np
import pytest

from lumivox_wakelab.vad import SileroVad, VadClosedError, VadInferenceError, VadInitializationError, _silero

from .test_artifacts import RecordingLogger


@dataclass
class Node:
    name: str
    type: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray[Any, Any]]] = []
        self.requested_outputs: list[list[str]] = []

    def get_inputs(self) -> list[Node]:
        return [
            Node("input", "tensor(float)", [None, None]),
            Node("state", "tensor(float)", [2, None, 128]),
            Node("sr", "tensor(int64)", []),
        ]

    def get_outputs(self) -> list[Node]:
        return [
            Node("output", "tensor(float)", [None, 1]),
            Node("stateN", "tensor(float)", [None, None, 128]),
        ]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
        self.requested_outputs.append(output_names)
        self.feeds.append(input_feed)
        probability = np.asarray([[0.25]], dtype=np.float32)
        next_state = input_feed["state"].copy()
        next_state[0, 0, 0] += 1.0
        return [probability, next_state]


def make_vad(session: FakeSession | None = None) -> tuple[SileroVad, FakeSession]:
    actual_session = session or FakeSession()
    return SileroVad(actual_session, ("CPUExecutionProvider",), RecordingLogger()), actual_session


def test_infer_uses_exact_native_contract_and_preserves_input() -> None:
    vad, session = make_vad()
    frame = np.linspace(-32768, 32767, 512, dtype=np.dtype("<i2"))
    original = frame.copy()

    probability = vad.infer(frame)

    assert probability == 0.25
    assert np.array_equal(frame, original)
    assert session.requested_outputs == [["output", "stateN"]]
    feed = session.feeds[0]
    assert set(feed) == {"input", "state", "sr"}
    assert feed["input"].shape == (1, 576)
    assert feed["input"].dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(feed["input"][:, :64], np.zeros((1, 64), dtype=np.float32))
    assert feed["input"][0, 64] == -1.0
    assert feed["sr"].shape == ()
    assert feed["sr"].dtype == np.dtype(np.int64)


def test_context_and_recurrent_state_propagate_then_reset() -> None:
    vad, session = make_vad()
    first = np.arange(512, dtype=np.dtype("<i2"))
    second = np.zeros(512, dtype=np.dtype("<i2"))

    vad.infer(first)
    vad.infer(second)
    np.testing.assert_array_equal(session.feeds[1]["input"][0, :64], first[-64:].astype(np.float32) / 32768.0)
    assert session.feeds[1]["state"][0, 0, 0] == 1.0

    vad.reset()
    vad.infer(second)
    np.testing.assert_array_equal(session.feeds[2]["input"][0, :64], np.zeros(64, dtype=np.float32))
    assert session.feeds[2]["state"][0, 0, 0] == 0.0


@pytest.mark.parametrize(
    ("frame", "error_type"),
    [
        ([0] * 512, TypeError),
        (np.zeros(0, dtype=np.dtype("<i2")), ValueError),
        (np.zeros((1, 512), dtype=np.dtype("<i2")), ValueError),
        (np.zeros(511, dtype=np.dtype("<i2")), ValueError),
        (np.zeros(513, dtype=np.dtype("<i2")), ValueError),
        (np.zeros(512, dtype=np.float32), TypeError),
        (np.zeros(512, dtype=np.dtype(">i2")), TypeError),
    ],
)
def test_invalid_frames_are_rejected(frame: object, error_type: type[Exception]) -> None:
    vad, _ = make_vad()
    with pytest.raises(error_type):
        vad.infer(frame)


class InvalidResultSession(FakeSession):
    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
        return [np.asarray([[float("nan")]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)]


def test_invalid_results_do_not_advance_stream_state() -> None:
    vad, _ = make_vad(InvalidResultSession())
    with pytest.raises(VadInferenceError):
        vad.infer(np.zeros(512, dtype=np.dtype("<i2")))


def test_close_is_idempotent_and_terminal() -> None:
    vad, _ = make_vad()
    vad.close()
    vad.close()

    with pytest.raises(VadClosedError):
        vad.infer(np.zeros(512, dtype=np.dtype("<i2")))
    with pytest.raises(VadClosedError):
        vad.reset()
    with pytest.raises(VadClosedError):
        vad.__enter__()


def test_context_manager_closes_after_exception() -> None:
    vad, _ = make_vad()
    with pytest.raises(RuntimeError), vad:
        raise RuntimeError("caller failure")
    with pytest.raises(VadClosedError):
        vad.reset()


class FakeSessionOptions:
    inter_op_num_threads = 0
    intra_op_num_threads = 0
    graph_optimization_level: object = None


def fake_onnxruntime(session: FakeSession, available: list[str] | None = None) -> SimpleNamespace:
    captured: dict[str, Any] = {}

    def inference_session(model: bytes, *, sess_options: FakeSessionOptions, providers: list[str]) -> FakeSession:
        captured.update(model=model, options=sess_options, providers=providers)
        return session

    return SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: available or ["CPUExecutionProvider"],
        InferenceSession=inference_session,
        captured=captured,
    )


def test_load_configures_cpu_validates_warms_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    runtime = fake_onnxruntime(session)
    monkeypatch.setattr(_silero, "read_verified_artifact", lambda path, manifest: b"verified")
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    vad = SileroVad.load("model.onnx", logger=RecordingLogger())

    assert runtime.captured["model"] == b"verified"
    assert runtime.captured["providers"] == ["CPUExecutionProvider"]
    options = runtime.captured["options"]
    assert options.inter_op_num_threads == 1
    assert options.intra_op_num_threads == 1
    assert options.graph_optimization_level == "all"
    # load() has completed one warm-up inference before the backend escapes.
    assert len(session.feeds) == 1
    vad.infer(np.zeros(512, dtype=np.dtype("<i2")))
    np.testing.assert_array_equal(session.feeds[1]["input"], np.zeros((1, 576), dtype=np.float32))
    assert session.feeds[1]["state"][0, 0, 0] == 0.0
    vad.close()


def test_load_rejects_unavailable_or_unexpected_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_silero, "read_verified_artifact", lambda path, manifest: b"verified")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(FakeSession(), ["AzureExecutionProvider"]))
    with pytest.raises(VadInitializationError, match="unavailable"):
        SileroVad.load("model.onnx", logger=RecordingLogger())

    session = FakeSession()
    session.get_providers = lambda: ["CPUExecutionProvider", "AzureExecutionProvider"]  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(session))
    with pytest.raises(VadInitializationError, match="unexpected providers"):
        SileroVad.load("model.onnx", logger=RecordingLogger())


def test_load_rejects_incompatible_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    session.get_inputs = lambda: [Node("legacy", "tensor(float)", [1, 576])]  # type: ignore[method-assign]
    monkeypatch.setattr(_silero, "read_verified_artifact", lambda path, manifest: b"verified")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(session))

    with pytest.raises(VadInitializationError, match="names"):
        SileroVad.load("model.onnx", logger=RecordingLogger())
