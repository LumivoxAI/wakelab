"""Voice activity detection backends and model preparation."""

import os
from pathlib import Path

from lumivox_core.logger import Logger

from lumivox_wakelab._artifacts import (
    ArtifactError,
    ArtifactMissingError,
    ArtifactDownloadError,
    ArtifactIntegrityError,
    ArtifactPublicationError,
    ensure_artifact,
)

from ._silero import (
    SILERO_VAD_MANIFEST,
    VadError,
    SileroVad,
    VadClosedError,
    VadInferenceError,
    VadInitializationError,
)
from ._backend import VadBackend


def ensure_silero_vad_model(
    model_root: str | os.PathLike[str],
    *,
    logger: Logger,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Prepare and return the verified pinned Silero VAD model path."""

    return ensure_artifact(
        model_root,
        SILERO_VAD_MANIFEST,
        logger=logger,
        allow_download=allow_download,
        timeout=timeout,
    )


__all__ = [
    "ArtifactDownloadError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactMissingError",
    "ArtifactPublicationError",
    "SileroVad",
    "VadClosedError",
    "VadError",
    "VadInferenceError",
    "VadInitializationError",
    "VadBackend",
    "ensure_silero_vad_model",
]
