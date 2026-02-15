"""Verified acquisition of externally distributed model artifacts."""

from __future__ import annotations

import os
import math
import fcntl
import hashlib
import tempfile
import urllib.error
import urllib.request
from typing import BinaryIO
from pathlib import Path
from dataclasses import dataclass

from lumivox_core.logger import Logger

_COPY_BUFFER_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Base class for model artifact failures."""


class ArtifactMissingError(ArtifactError):
    """Raised when a required artifact is not cached."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when artifact bytes do not match their manifest."""


class ArtifactDownloadError(ArtifactError):
    """Raised when an artifact cannot be downloaded completely."""


class ArtifactPublicationError(ArtifactError):
    """Raised when a verified artifact cannot be published to the cache."""


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Immutable description of one externally hosted artifact."""

    name: str
    version: str
    filename: str
    url: str
    size: int
    sha256: str

    def path_under(self, model_root: str | os.PathLike[str]) -> Path:
        return Path(model_root) / self.name / self.version / self.filename


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BUFFER_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_artifact_file(path: str | os.PathLike[str], manifest: ArtifactManifest) -> Path:
    """Verify an existing artifact against code-owned size and digest metadata."""

    artifact_path = Path(path)
    try:
        size, digest = _digest_file(artifact_path)
    except FileNotFoundError as exc:
        raise ArtifactMissingError(f"Model artifact is missing: {artifact_path}") from exc
    except OSError as exc:
        raise ArtifactIntegrityError(f"Cannot read model artifact: {artifact_path}") from exc

    if size != manifest.size or digest != manifest.sha256:
        raise ArtifactIntegrityError(
            f"Model artifact failed integrity verification: {artifact_path} "
            f"(expected size {manifest.size} and SHA-256 {manifest.sha256}, "
            f"got size {size} and SHA-256 {digest})"
        )
    return artifact_path


def read_verified_artifact(path: str | os.PathLike[str], manifest: ArtifactManifest) -> bytes:
    """Read and verify bytes for consumers that must avoid a verify/open race."""

    artifact_path = Path(path)
    try:
        contents = artifact_path.read_bytes()
    except FileNotFoundError as exc:
        raise ArtifactMissingError(f"Model artifact is missing: {artifact_path}") from exc
    except OSError as exc:
        raise ArtifactIntegrityError(f"Cannot read model artifact: {artifact_path}") from exc
    digest = hashlib.sha256(contents).hexdigest()
    if len(contents) != manifest.size or digest != manifest.sha256:
        raise ArtifactIntegrityError(
            f"Model artifact failed integrity verification: {artifact_path} "
            f"(expected size {manifest.size} and SHA-256 {manifest.sha256}, "
            f"got size {len(contents)} and SHA-256 {digest})"
        )
    return contents


def _sidecar_matches(path: Path, digest: str) -> bool:
    try:
        return path.read_text(encoding="ascii").strip() == digest
    except (OSError, UnicodeError):
        return False


def _replace_sidecar(path: Path, digest: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"{digest}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _repair_sidecar(path: Path, manifest: ArtifactManifest, logger: Logger) -> None:
    sidecar_path = path.with_name(f"{path.name}.sha256")
    if _sidecar_matches(sidecar_path, manifest.sha256):
        return
    try:
        _replace_sidecar(sidecar_path, manifest.sha256)
    except OSError as exc:
        logger.warning("artifact_sidecar_repair_failed", path=str(sidecar_path), error=str(exc))
    else:
        logger.debug("artifact_sidecar_repaired", path=str(sidecar_path))


def _verified_if_present(path: Path, manifest: ArtifactManifest) -> bool:
    try:
        verify_artifact_file(path, manifest)
    except ArtifactError:
        return False
    return True


def _download_to(path: Path, manifest: ArtifactManifest, timeout: float) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(manifest.url, timeout=timeout) as response, path.open("wb") as stream:
            status = getattr(response, "status", 200)
            if status < 200 or status >= 300:
                raise ArtifactDownloadError(f"Artifact server returned HTTP status {status}")
            while chunk := response.read(_COPY_BUFFER_SIZE):
                size += len(chunk)
                if size > manifest.size:
                    raise ArtifactIntegrityError(f"Downloaded artifact exceeds expected size {manifest.size}")
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except (ArtifactDownloadError, ArtifactIntegrityError):
        raise
    except Exception as exc:
        raise ArtifactDownloadError(f"Failed to download artifact from {manifest.url}") from exc

    actual_digest = digest.hexdigest()
    if size != manifest.size or actual_digest != manifest.sha256:
        raise ArtifactIntegrityError(
            f"Downloaded artifact failed integrity verification "
            f"(expected size {manifest.size} and SHA-256 {manifest.sha256}, "
            f"got size {size} and SHA-256 {actual_digest})"
        )


def ensure_artifact(
    model_root: str | os.PathLike[str],
    manifest: ArtifactManifest,
    *,
    logger: Logger,
    allow_download: bool,
    timeout: float,
) -> Path:
    """Return a verified cached artifact, downloading it atomically if allowed."""

    if not isinstance(timeout, int | float) or isinstance(timeout, bool):
        raise ValueError("timeout must be a positive finite number")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a positive finite number")

    bound_logger = logger.bind(module="wakelab", component="artifacts")
    artifact_path = manifest.path_under(model_root)
    initially_valid = _verified_if_present(artifact_path, manifest)
    if initially_valid:
        bound_logger.debug("artifact_cache_hit", path=str(artifact_path))
        if _sidecar_matches(artifact_path.with_name(f"{artifact_path.name}.sha256"), manifest.sha256):
            return artifact_path
    elif not allow_download and not artifact_path.parent.exists():
        if artifact_path.exists():
            bound_logger.error("artifact_cache_corrupt", path=str(artifact_path))
            raise ArtifactIntegrityError(f"Cached model artifact is corrupt: {artifact_path}")
        bound_logger.error("artifact_cache_missing", path=str(artifact_path))
        raise ArtifactMissingError(f"Model artifact is not cached: {artifact_path}")

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = artifact_path.with_name(f".{artifact_path.name}.lock")
        lock_stream: BinaryIO = lock_path.open("a+b")
    except OSError as exc:
        if initially_valid:
            bound_logger.warning("artifact_sidecar_repair_failed", path=str(artifact_path), error=str(exc))
            return artifact_path
        raise ArtifactPublicationError(f"Cannot prepare artifact cache: {artifact_path.parent}") from exc

    with lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if initially_valid:
                bound_logger.warning("artifact_sidecar_repair_failed", path=str(artifact_path), error=str(exc))
                return artifact_path
            raise ArtifactPublicationError(f"Cannot lock artifact cache: {artifact_path}") from exc

        if _verified_if_present(artifact_path, manifest):
            bound_logger.debug("artifact_cache_hit_after_lock", path=str(artifact_path))
            _repair_sidecar(artifact_path, manifest, bound_logger)
            return artifact_path

        if not allow_download:
            if artifact_path.exists():
                raise ArtifactIntegrityError(f"Cached model artifact is corrupt: {artifact_path}")
            raise ArtifactMissingError(f"Model artifact is not cached: {artifact_path}")

        bound_logger.info("artifact_download_started", url=manifest.url, path=str(artifact_path))
        descriptor = -1
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{artifact_path.name}.", suffix=".download", dir=artifact_path.parent
            )
            os.close(descriptor)
        except OSError as exc:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise ArtifactPublicationError(f"Cannot stage model artifact: {artifact_path}") from exc
        assert temporary_name is not None
        temporary_path = Path(temporary_name)
        try:
            _download_to(temporary_path, manifest, timeout_value)
            try:
                os.replace(temporary_path, artifact_path)
            except OSError as exc:
                raise ArtifactPublicationError(f"Cannot publish model artifact: {artifact_path}") from exc
        except BaseException as exc:
            temporary_path.unlink(missing_ok=True)
            bound_logger.error("artifact_acquisition_failed", path=str(artifact_path), error=str(exc))
            raise

        _repair_sidecar(artifact_path, manifest, bound_logger)
        bound_logger.info("artifact_download_completed", path=str(artifact_path))
        return artifact_path
