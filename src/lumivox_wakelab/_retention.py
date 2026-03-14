"""Conservative retained-audio bound shared by all processor constructors."""

from .stream import StreamConfig, _validate_positive_integer


def required_retained_audio_samples(
    config: StreamConfig,
    vad_frame_samples: int,
    wake_frame_samples: int,
    /,
) -> int:
    """Return the maximum simultaneously unsafe and unfinalized PCM."""

    if not isinstance(config, StreamConfig):
        raise TypeError("config must be a StreamConfig")
    _validate_positive_integer("vad_frame_samples", vad_frame_samples)
    _validate_positive_integer("wake_frame_samples", wake_frame_samples)

    vad = config.vad_policy
    wake = config.wake_policy
    speech_pending = vad.left_padding_samples + _round_up(vad.minimum_speech_samples, vad_frame_samples)
    silence_pending = _round_up(vad.minimum_silence_samples, vad_frame_samples)
    wake_unsafe = (
        wake.pre_roll_samples + wake.consecutive_score_count * wake_frame_samples + wake.silence_bridge_samples
    )
    return wake_unsafe + max(speech_pending, silence_pending)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple
