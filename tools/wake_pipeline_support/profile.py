"""Strict version-one profile persistence and real processor construction."""

from __future__ import annotations

import os
import json
import hashlib
import tempfile
from typing import cast
from pathlib import Path
from dataclasses import asdict, dataclass
from collections.abc import Mapping

from lumivox_core.logger import Logger

from lumivox_wakelab import StreamConfig, VadPolicyConfig, WakePolicyConfig, WakeStreamProcessor
from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model
from lumivox_wakelab.wakeword import OpenWakeWord, ensure_openwakeword_feature_models
from lumivox_wakelab._retention import required_retained_audio_samples

_ROOT_FIELDS = {
    "schema_version",
    "name",
    "model_root",
    "allow_download",
    "classifier_path",
    "classifier_sha256",
    "vad_policy",
    "wake_policy",
    "max_retained_audio_samples",
}
_VAD_FIELDS = {
    "speech_threshold",
    "silence_threshold",
    "minimum_speech_samples",
    "minimum_silence_samples",
    "left_padding_samples",
    "right_padding_samples",
}
_WAKE_FIELDS = {"score_threshold", "consecutive_score_count", "silence_bridge_samples", "pre_roll_samples"}
_SILERO_FRAME_SAMPLES = 512
_OPENWAKEWORD_FRAME_SAMPLES = 1_280


@dataclass(frozen=True, slots=True)
class ProfileDraft:
    """Editable profile values whose paths remain relative strings."""

    name: str
    model_root: str
    allow_download: bool
    classifier_path: str
    classifier_sha256: str
    vad_policy: VadPolicyConfig
    wake_policy: WakePolicyConfig
    max_retained_audio_samples: int


@dataclass(frozen=True, slots=True)
class MicrophoneProfile:
    """Validated profile with paths resolved from its server-local JSON file."""

    draft: ProfileDraft
    path: Path
    model_root: Path
    classifier_path: Path
    stream_config: StreamConfig

    @property
    def name(self) -> str:
        return self.draft.name

    @property
    def allow_download(self) -> bool:
        return self.draft.allow_download

    @property
    def classifier_sha256(self) -> str:
        return self.draft.classifier_sha256


@dataclass(frozen=True, slots=True)
class BuiltProcessor:
    processor: WakeStreamProcessor
    providers: tuple[str, ...]


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_fields(mapping: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = expected - mapping.keys()
    unknown = mapping.keys() - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _required(mapping: Mapping[str, object], name: str, expected: type[object]) -> object:
    value = mapping[name]
    if not isinstance(value, expected):
        raise ValueError(f"profile field {name!r} must be {expected.__name__}")
    return value


def _policy(mapping: Mapping[str, object], policy_type: type[object]) -> object:
    try:
        return policy_type(**mapping)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {policy_type.__name__}: {error}") from error


def parse_profile(value: object) -> ProfileDraft:
    root = _object(value, "profile")
    if root.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")
    _exact_fields(root, _ROOT_FIELDS, "profile")
    name = cast(str, _required(root, "name", str))
    model_root = cast(str, _required(root, "model_root", str))
    classifier_path = cast(str, _required(root, "classifier_path", str))
    if not name.strip():
        raise ValueError("profile field 'name' must not be blank")
    if not model_root.strip() or not classifier_path.strip():
        raise ValueError("profile path fields must not be blank")
    digest = cast(str, _required(root, "classifier_sha256", str))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("profile field 'classifier_sha256' must be a lowercase SHA-256 digest")

    vad_mapping = _object(_required(root, "vad_policy", dict), "vad_policy")
    wake_mapping = _object(_required(root, "wake_policy", dict), "wake_policy")
    _exact_fields(vad_mapping, _VAD_FIELDS, "vad_policy")
    _exact_fields(wake_mapping, _WAKE_FIELDS, "wake_policy")
    vad = cast(VadPolicyConfig, _policy(vad_mapping, VadPolicyConfig))
    wake = cast(WakePolicyConfig, _policy(wake_mapping, WakePolicyConfig))
    maximum = cast(int, _required(root, "max_retained_audio_samples", int))
    config = StreamConfig(vad, wake, maximum)
    required = required_retained_audio_samples(
        config,
        _SILERO_FRAME_SAMPLES,
        _OPENWAKEWORD_FRAME_SAMPLES,
    )
    if maximum < required:
        raise ValueError(f"max_retained_audio_samples must be at least {required} for the real backends")
    return ProfileDraft(
        name,
        model_root,
        cast(bool, _required(root, "allow_download", bool)),
        classifier_path,
        digest,
        vad,
        wake,
        maximum,
    )


def resolve_profile(draft: ProfileDraft, path: Path) -> MicrophoneProfile:
    profile_path = path.expanduser().resolve()
    base = profile_path.parent
    stream_config = StreamConfig(draft.vad_policy, draft.wake_policy, draft.max_retained_audio_samples)
    return MicrophoneProfile(
        draft,
        profile_path,
        (base / draft.model_root).resolve(),
        (base / draft.classifier_path).resolve(),
        stream_config,
    )


def load_profile(path: Path) -> MicrophoneProfile:
    profile_path = path.expanduser().resolve()
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read profile {profile_path}: {error}") from error
    return resolve_profile(parse_profile(raw), profile_path)


def profile_json(draft: ProfileDraft) -> str:
    value: dict[str, object] = {
        "schema_version": 1,
        "name": draft.name,
        "model_root": draft.model_root,
        "allow_download": draft.allow_download,
        "classifier_path": draft.classifier_path,
        "classifier_sha256": draft.classifier_sha256,
        "vad_policy": asdict(draft.vad_policy),
        "wake_policy": asdict(draft.wake_policy),
        "max_retained_audio_samples": draft.max_retained_audio_samples,
    }
    return json.dumps(value, indent=2, ensure_ascii=True) + "\n"


def save_profile(path: Path, draft: ProfileDraft, *, replace: bool = False) -> MicrophoneProfile:
    destination = path.expanduser().resolve()
    if not destination.parent.is_dir():
        raise ValueError(f"profile directory does not exist: {destination.parent}")
    # Validate the complete serialized contract before touching the destination.
    validated = parse_profile(json.loads(profile_json(draft)))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(profile_json(validated))
            temporary.flush()
            os.fsync(temporary.fileno())
        if replace:
            os.replace(temporary_path, destination)
            temporary_path = None
        else:
            os.link(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return resolve_profile(validated, destination)


def verify_classifier(profile: MicrophoneProfile) -> bytes:
    """Return the exact classifier snapshot whose digest was verified."""

    try:
        classifier_bytes = profile.classifier_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read classifier {profile.classifier_path}: {error}") from error
    actual = hashlib.sha256(classifier_bytes).hexdigest()
    if actual != profile.classifier_sha256:
        raise ValueError(
            f"classifier SHA-256 mismatch for {profile.classifier_path}: "
            f"expected {profile.classifier_sha256}, got {actual}"
        )
    return classifier_bytes


def build_processor(
    profile: MicrophoneProfile,
    logger: Logger,
    *,
    diagnostic_event_capacity: int = 0,
) -> BuiltProcessor:
    classifier_bytes = verify_classifier(profile)
    vad_path = ensure_silero_vad_model(profile.model_root, logger=logger, allow_download=profile.allow_download)
    feature_dir = ensure_openwakeword_feature_models(
        profile.model_root, logger=logger, allow_download=profile.allow_download
    )
    vad = SileroVad.load(vad_path, logger=logger)
    try:
        wake = OpenWakeWord.load(
            feature_dir,
            profile.classifier_path,
            logger=logger,
            classifier_bytes=classifier_bytes,
        )
    except Exception:
        vad.close()
        raise
    providers = tuple(dict.fromkeys((*vad.active_providers, *wake.active_providers)))
    try:
        processor = WakeStreamProcessor(
            profile.stream_config,
            vad,
            wake,
            logger=logger,
            diagnostic_event_capacity=diagnostic_event_capacity,
        )
    except Exception:
        vad.close()
        wake.close()
        raise
    return BuiltProcessor(processor, providers)
