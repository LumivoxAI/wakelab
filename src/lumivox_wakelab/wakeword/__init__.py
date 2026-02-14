"""Wake-word inference backends and feature-model preparation."""

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

from ._openwakeword import (
    EMBEDDING_MANIFEST,
    MELSPECTROGRAM_MANIFEST,
    OpenWakeWord,
    WakeWordError,
    WakeWordClosedError,
    WakeWordInferenceError,
    WakeWordInitializationError,
)


def ensure_openwakeword_feature_models(
    model_root: str | os.PathLike[str],
    *,
    logger: Logger,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Prepare and return the directory containing both verified feature models."""

    paths = [
        ensure_artifact(
            model_root,
            manifest,
            logger=logger,
            allow_download=allow_download,
            timeout=timeout,
        )
        for manifest in (MELSPECTROGRAM_MANIFEST, EMBEDDING_MANIFEST)
    ]
    assert paths[0].parent == paths[1].parent
    return paths[0].parent


__all__ = [
    "ArtifactDownloadError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactMissingError",
    "ArtifactPublicationError",
    "OpenWakeWord",
    "WakeWordClosedError",
    "WakeWordError",
    "WakeWordInferenceError",
    "WakeWordInitializationError",
    "ensure_openwakeword_feature_models",
]
