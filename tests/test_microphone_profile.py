from __future__ import annotations

import json
import hashlib
from typing import cast
from pathlib import Path
from collections.abc import Mapping

import pytest

from tools.wake_pipeline_support.profile import (
    load_profile,
    save_profile,
    parse_profile,
    verify_classifier,
)


def profile_value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test",
        "model_root": "models",
        "allow_download": False,
        "classifier_path": "classifier.onnx",
        "classifier_sha256": "0" * 64,
        "vad_policy": {
            "speech_threshold": 0.6,
            "silence_threshold": 0.4,
            "minimum_speech_samples": 4,
            "minimum_silence_samples": 4,
            "left_padding_samples": 0,
            "right_padding_samples": 0,
        },
        "wake_policy": {
            "score_threshold": 0.7,
            "consecutive_score_count": 1,
            "silence_bridge_samples": 4,
            "pre_roll_samples": 2,
        },
        "max_retained_audio_samples": 4096,
    }


def test_profile_round_trip_is_deterministic_and_paths_are_relative_to_file(tmp_path: Path) -> None:
    draft = parse_profile(profile_value())
    destination = tmp_path / "profile.json"

    resolved = save_profile(destination, draft)
    loaded = load_profile(destination)

    assert loaded == resolved
    assert loaded.model_root == tmp_path / "models"
    assert loaded.classifier_path == tmp_path / "classifier.onnx"
    first = destination.read_bytes()
    save_profile(destination, draft, replace=True)
    assert destination.read_bytes() == first
    assert first.endswith(b"\n")


def test_profile_refuses_unknown_fields_and_invalid_cross_field_values() -> None:
    value = profile_value()
    value["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        parse_profile(value)

    value = profile_value()
    vad = dict(cast(Mapping[str, object], value["vad_policy"]))
    vad["silence_threshold"] = 0.8
    value["vad_policy"] = vad
    with pytest.raises(ValueError, match="silence_threshold"):
        parse_profile(value)

    value = profile_value()
    value["max_retained_audio_samples"] = 100
    with pytest.raises(ValueError, match="real backends"):
        parse_profile(value)


def test_profile_no_replace_is_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    destination.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_profile(destination, parse_profile(profile_value()))
    assert destination.read_text(encoding="utf-8") == "existing"


def test_classifier_digest_is_verified_incrementally(tmp_path: Path) -> None:
    classifier = tmp_path / "classifier.onnx"
    classifier.write_bytes(b"classifier")
    value = profile_value()
    value["classifier_sha256"] = hashlib.sha256(b"classifier").hexdigest()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(value), encoding="utf-8")

    verify_classifier(load_profile(profile_path))
    classifier.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_classifier(load_profile(profile_path))
