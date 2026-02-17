from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model

from .test_artifacts import RecordingLogger


@pytest.mark.model
def test_real_model_acquisition_inference_and_offline_reuse(tmp_path: Path) -> None:
    configured_root = os.environ.get("WAKELAB_MODEL_ROOT")
    model_root = Path(configured_root) if configured_root else tmp_path
    logger = RecordingLogger()
    model_path = ensure_silero_vad_model(model_root, logger=logger, allow_download=True)
    assert model_path.relative_to(model_root) == Path("silero-vad/v6.2.1/silero_vad.onnx")
    assert ensure_silero_vad_model(model_root, logger=logger, allow_download=False) == model_path

    silence = np.zeros(512, dtype=np.dtype("<i2"))
    signal = np.arange(-256, 256, dtype=np.dtype("<i2"))
    vad = SileroVad.load(model_path, logger=logger)
    try:
        assert vad.active_providers == ("CPUExecutionProvider",)
        assert vad.sample_rate == 16_000
        assert vad.frame_samples == 512
        assert vad.model_version == "6.2.1"
        silence_probability = vad.infer(silence)
        signal_probability = vad.infer(signal)
        assert silence_probability == pytest.approx(0.0016697943, abs=1e-7)
        assert signal_probability == pytest.approx(0.18514708, abs=1e-6)
        vad.reset()
        assert vad.infer(silence) == pytest.approx(silence_probability, abs=1e-7)
    finally:
        vad.close()

    reloaded = SileroVad.load(model_path, logger=logger)
    try:
        assert reloaded.infer(silence) == pytest.approx(silence_probability, abs=1e-7)
    finally:
        reloaded.close()
