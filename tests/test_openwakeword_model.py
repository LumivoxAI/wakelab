from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lumivox_wakelab.wakeword import OpenWakeWord, ensure_openwakeword_feature_models

from .test_artifacts import RecordingLogger


@pytest.mark.model
def test_real_openwakeword_acquisition_inference_reset_and_reload(tmp_path: Path) -> None:
    configured_classifier = os.environ.get("WAKELAB_OPENWAKEWORD_CLASSIFIER")
    if not configured_classifier:
        pytest.skip("WAKELAB_OPENWAKEWORD_CLASSIFIER does not select a trusted local binary classifier")

    configured_root = os.environ.get("WAKELAB_MODEL_ROOT")
    model_root = Path(configured_root) if configured_root else tmp_path
    classifier_path = Path(configured_classifier)
    logger = RecordingLogger()
    feature_dir = ensure_openwakeword_feature_models(model_root, logger=logger, allow_download=True)
    assert feature_dir.relative_to(model_root) == Path("openwakeword-features/v0.5.1")
    assert ensure_openwakeword_feature_models(model_root, logger=logger, allow_download=False) == feature_dir

    silence = np.zeros(1_280, dtype=np.dtype("<i2"))
    signal = np.arange(-640, 640, dtype=np.dtype("<i2"))
    backend = OpenWakeWord.load(feature_dir, classifier_path, logger=logger)
    try:
        assert backend.active_providers == ("CPUExecutionProvider",)
        assert backend.sample_rate == 16_000
        assert backend.frame_samples == 1_280
        assert backend.feature_version == "0.5.1"
        assert 1 <= backend.classifier_history_frames <= 120
        silence_score = backend.infer(silence)
        signal_score = backend.infer(signal)
        assert silence_score == pytest.approx(0.0, abs=1e-7)
        assert signal_score == pytest.approx(0.0, abs=1e-7)
        backend.reset()
        assert backend.infer(silence) == pytest.approx(silence_score, abs=1e-7)
    finally:
        backend.close()

    reloaded = OpenWakeWord.load(feature_dir, classifier_path, logger=logger)
    try:
        assert reloaded.infer(silence) == pytest.approx(silence_score, abs=1e-7)
    finally:
        reloaded.close()
