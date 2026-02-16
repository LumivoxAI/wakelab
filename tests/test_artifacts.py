from __future__ import annotations

import hashlib
from typing import Any, Self
from pathlib import Path

import pytest

from lumivox_wakelab._artifacts import (
    ArtifactManifest,
    ArtifactMissingError,
    ArtifactDownloadError,
    ArtifactIntegrityError,
    ArtifactPublicationError,
    ensure_artifact,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def bind(self, **new_values: Any) -> Self:
        return self

    def debug(self, event: str, **kwargs: Any) -> None:
        self.events.append(("debug", event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append(("info", event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.events.append(("warning", event, kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self.events.append(("error", event, kwargs))

    def critical(self, event: str, **kwargs: Any) -> None:
        self.events.append(("critical", event, kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.events.append(("exception", event, kwargs))


class BytesResponse:
    status = 200

    def __init__(self, payload: bytes, *, failure: Exception | None = None) -> None:
        self._payload = payload
        self._failure = failure
        self._read = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._failure is not None:
            raise self._failure
        if self._read:
            return b""
        self._read = True
        return self._payload


@pytest.fixture
def manifest() -> ArtifactManifest:
    payload = b"verified model bytes"
    return ArtifactManifest(
        name="model",
        version="1",
        filename="model.onnx",
        url="https://example.invalid/model.onnx",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_valid_cache_is_reused_and_sidecar_is_repaired(
    tmp_path: Path, manifest: ArtifactManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"verified model bytes")
    sidecar = model_path.with_name(f"{model_path.name}.sha256")
    sidecar.write_text("stale\n", encoding="ascii")

    def fail_urlopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("cache hit attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    result = ensure_artifact(
        tmp_path,
        manifest,
        logger=RecordingLogger(),
        allow_download=False,
        timeout=1.0,
    )

    assert result == model_path
    assert sidecar.read_text(encoding="ascii") == f"{manifest.sha256}\n"


def test_offline_missing_and_corrupt_cache_are_not_modified(tmp_path: Path, manifest: ArtifactManifest) -> None:
    logger = RecordingLogger()
    with pytest.raises(ArtifactMissingError):
        ensure_artifact(tmp_path, manifest, logger=logger, allow_download=False, timeout=1.0)
    assert not tmp_path.joinpath("model").exists()

    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"corrupt")
    sidecar = model_path.with_name(f"{model_path.name}.sha256")
    sidecar.write_text(f"{manifest.sha256}\n", encoding="ascii")
    before = model_path.read_bytes()

    with pytest.raises(ArtifactIntegrityError):
        ensure_artifact(tmp_path, manifest, logger=logger, allow_download=False, timeout=1.0)
    assert model_path.read_bytes() == before


def test_download_replaces_corrupt_cache_atomically(
    tmp_path: Path, manifest: ArtifactManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: BytesResponse(b"verified model bytes"),
    )

    result = ensure_artifact(
        tmp_path,
        manifest,
        logger=RecordingLogger(),
        allow_download=True,
        timeout=1.0,
    )

    assert result.read_bytes() == b"verified model bytes"
    assert not list(model_path.parent.glob("*.download"))


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(tmp_path: Path, manifest: ArtifactManifest, timeout: float) -> None:
    with pytest.raises(ValueError):
        ensure_artifact(
            tmp_path,
            manifest,
            logger=RecordingLogger(),
            allow_download=False,
            timeout=timeout,
        )


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (BytesResponse(b"wrong"), ArtifactIntegrityError),
        (BytesResponse(b"", failure=TimeoutError()), ArtifactDownloadError),
    ],
)
def test_failed_download_keeps_existing_cache_and_cleans_temporary_file(
    tmp_path: Path,
    manifest: ArtifactManifest,
    monkeypatch: pytest.MonkeyPatch,
    response: BytesResponse,
    error_type: type[Exception],
) -> None:
    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"old corrupt bytes")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)

    with pytest.raises(error_type):
        ensure_artifact(
            tmp_path,
            manifest,
            logger=RecordingLogger(),
            allow_download=True,
            timeout=1.0,
        )

    assert model_path.read_bytes() == b"old corrupt bytes"
    assert not list(model_path.parent.glob("*.download"))


def test_offline_caller_rechecks_cache_after_lock(
    tmp_path: Path, manifest: ArtifactManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)

    def publish_while_waiting(descriptor: int, operation: int) -> None:
        model_path.write_bytes(b"verified model bytes")

    monkeypatch.setattr("fcntl.flock", publish_while_waiting)
    result = ensure_artifact(
        tmp_path,
        manifest,
        logger=RecordingLogger(),
        allow_download=False,
        timeout=1.0,
    )

    assert result == model_path


def test_publication_failure_keeps_old_cache_and_cleans_temporary_file(
    tmp_path: Path, manifest: ArtifactManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = manifest.path_under(tmp_path)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"old corrupt bytes")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: BytesResponse(b"verified model bytes"),
    )

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("full")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(ArtifactPublicationError):
        ensure_artifact(
            tmp_path,
            manifest,
            logger=RecordingLogger(),
            allow_download=True,
            timeout=1.0,
        )

    assert model_path.read_bytes() == b"old corrupt bytes"
    assert not list(model_path.parent.glob("*.download"))
