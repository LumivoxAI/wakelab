"""Localized parameter help for the wake-pipeline workbench."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

HelpLanguage = Literal["en", "ru"]


@dataclass(frozen=True, slots=True)
class HelpEntry:
    title: str
    body: str


_HELP: dict[HelpLanguage, dict[str, HelpEntry]] = {
    "en": {
        "vad_speech_threshold": HelpEntry(
            "VAD speech threshold",
            "A Silero VAD probability at or above this value starts or continues a speech candidate. "
            "It must be greater than the silence threshold.",
        ),
        "vad_silence_threshold": HelpEntry(
            "VAD silence threshold",
            "A Silero VAD probability below this value starts the silence debounce while speech is active. "
            "The gap between the two VAD thresholds provides hysteresis and prevents rapid state changes.",
        ),
        "wake_threshold": HelpEntry(
            "Wake threshold",
            "The minimum wake-word classifier score that counts as a hit. Activation still requires the configured "
            "number of consecutive hits.",
        ),
        "wake_streak": HelpEntry(
            "Wake streak",
            "The number of consecutive wake scores at or above the wake threshold required for activation. "
            "A larger value usually reduces false activations but increases detection delay.",
        ),
        "minimum_speech": HelpEntry(
            "Minimum speech samples",
            "How many consecutive samples of speech evidence are required before VAD confirms speech. Shorter "
            "candidates are rejected. The value is an exact 16 kHz sample count.",
        ),
        "minimum_silence": HelpEntry(
            "Minimum silence samples",
            "How many consecutive samples below the silence threshold are required before VAD ends speech. "
            "The value is an exact 16 kHz sample count.",
        ),
        "left_padding": HelpEntry(
            "Left padding samples",
            "Audio included before a confirmed VAD speech start, so initial phonemes are not lost. The value is an "
            "exact 16 kHz sample count.",
        ),
        "right_padding": HelpEntry(
            "Right padding samples",
            "Audio included after a confirmed VAD speech end, so final phonemes are not lost. The value is an exact "
            "16 kHz sample count.",
        ),
        "silence_bridge": HelpEntry(
            "Silence bridge samples",
            "The maximum short silence the wake-word evaluator may cross while resolving one candidate. This is "
            "separate from VAD's minimum silence. The value is an exact 16 kHz sample count.",
        ),
        "pre_roll": HelpEntry(
            "Pre-roll samples",
            "Audio retained before the first qualifying wake score and marked active after detection. It must cover "
            "the classifier's measured detection delay. The value is an exact 16 kHz sample count.",
        ),
        "maximum_retained": HelpEntry(
            "Maximum retained samples",
            "The hard bound on audio held while metadata and wake activation are being resolved. It must be large "
            "enough for the configured policy windows. The value is an exact 16 kHz sample count.",
        ),
    },
    "ru": {
        "vad_speech_threshold": HelpEntry(
            "Порог речи VAD",
            "Вероятность Silero VAD не ниже этого значения запускает или продолжает кандидата речи. Значение "
            "должно быть выше порога тишины.",
        ),
        "vad_silence_threshold": HelpEntry(
            "Порог тишины VAD",
            "Вероятность Silero VAD ниже этого значения запускает подтверждение тишины во время речи. Разница между "
            "двумя порогами создает гистерезис и не дает состоянию часто переключаться.",
        ),
        "wake_threshold": HelpEntry(
            "Порог wake word",
            "Минимальная оценка классификатора ключевого слова, которая считается попаданием. Для активации все "
            "равно требуется заданное число последовательных попаданий.",
        ),
        "wake_streak": HelpEntry(
            "Серия wake word",
            "Число последовательных оценок не ниже порога wake word, необходимое для активации. Большее значение "
            "обычно уменьшает ложные активации, но увеличивает задержку обнаружения.",
        ),
        "minimum_speech": HelpEntry(
            "Минимум сэмплов речи",
            "Число последовательных сэмплов с признаками речи, необходимое VAD для подтверждения речи. Более короткие "
            "кандидаты отклоняются. Значение задается точным числом сэмплов при 16 кГц.",
        ),
        "minimum_silence": HelpEntry(
            "Минимум сэмплов тишины",
            "Число последовательных сэмплов ниже порога тишины, необходимое VAD для завершения речи. Значение задается "
            "точным числом сэмплов при 16 кГц.",
        ),
        "left_padding": HelpEntry(
            "Левый отступ в сэмплах",
            "Аудио, добавляемое перед подтвержденным началом речи VAD, чтобы не потерять первые фонемы. Значение "
            "задается точным числом сэмплов при 16 кГц.",
        ),
        "right_padding": HelpEntry(
            "Правый отступ в сэмплах",
            "Аудио, добавляемое после подтвержденного конца речи VAD, чтобы не потерять последние фонемы. Значение "
            "задается точным числом сэмплов при 16 кГц.",
        ),
        "silence_bridge": HelpEntry(
            "Мост тишины в сэмплах",
            "Максимальная короткая тишина, через которую обработчик wake word может продолжить оценку одного "
            "кандидата. Этот параметр не равен минимуму тишины VAD и задается точным числом сэмплов при 16 кГц.",
        ),
        "pre_roll": HelpEntry(
            "Pre-roll в сэмплах",
            "Аудио перед первой подходящей оценкой wake word, которое сохраняется и после обнаружения помечается "
            "активным. Интервал должен покрывать измеренную задержку классификатора и задается сэмплами при 16 кГц.",
        ),
        "maximum_retained": HelpEntry(
            "Максимум сохраняемых сэмплов",
            "Жесткое ограничение объема аудио, удерживаемого до определения метаданных и активации. Оно должно "
            "вмещать настроенные временные окна и задается точным числом сэмплов при 16 кГц.",
        ),
    },
}


def language_from_locale(locale_name: object) -> HelpLanguage:
    """Map a browser locale to one of the supported help languages."""
    if isinstance(locale_name, str) and locale_name.replace("_", "-").lower().split("-", 1)[0] == "ru":
        return "ru"
    return "en"


def parameter_help(key: str, language: HelpLanguage) -> HelpEntry:
    return _HELP[language][key]
