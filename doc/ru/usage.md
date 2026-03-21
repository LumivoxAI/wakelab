# Использование библиотеки

## Контракт аудио

`InputChunk.samples` должен быть доступным для записи, непустым, одномерным
`numpy.ndarray` с точным dtype `numpy.dtype("<i2")`. Массив содержит моно PCM с
частотой 16 000 Гц. Байты, float-аудио, несколько каналов и другая частота
дискретизации не поддерживаются.

Длина входных блоков произвольна. Не нужно заранее разбивать звук по размеру кадров
моделей.

## Создание процессора

Получение моделей и загрузка бэкендов являются синхронными операциями подготовки.
Значения ниже взяты из некалиброванного демо-профиля репозитория. Это пример, а не
настройки по умолчанию для production.

```python
from pathlib import Path

from lumivox_core.logger import Logger
from lumivox_wakelab import (
    StreamConfig,
    VadPolicyConfig,
    WakePolicyConfig,
    WakeStreamProcessor,
)
from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model
from lumivox_wakelab.wakeword import (
    OpenWakeWord,
    ensure_openwakeword_feature_models,
)

# Приложение должно настроить lumivox-core и предоставить этот логгер.
logger: Logger = application_logger
model_root = Path("model-data")
classifier_path = Path("models/my-wake-word.onnx")

vad_path = ensure_silero_vad_model(model_root, logger=logger)
feature_dir = ensure_openwakeword_feature_models(model_root, logger=logger)

vad = SileroVad.load(vad_path, logger=logger)
try:
    wake = OpenWakeWord.load(feature_dir, classifier_path, logger=logger)
except Exception:
    vad.close()
    raise

config = StreamConfig(
    vad_policy=VadPolicyConfig(
        speech_threshold=0.5,
        silence_threshold=0.35,
        minimum_speech_samples=4_000,
        minimum_silence_samples=1_600,
        left_padding_samples=480,
        right_padding_samples=480,
    ),
    wake_policy=WakePolicyConfig(
        score_threshold=0.4,
        consecutive_score_count=3,
        silence_bridge_samples=8_000,
        pre_roll_samples=24_000,
    ),
    max_retained_audio_samples=160_000,
)

try:
    processor = WakeStreamProcessor(config, vad, wake, logger=logger)
except Exception:
    vad.close()
    wake.close()
    raise
```

Для обычного создания используйте `SileroVad.load()` и `OpenWakeWord.load()`.
Низкоуровневые конструкторы этих классов не являются точками входа приложения.

## Обработка аудио

```python
import numpy as np

from lumivox_wakelab import InputChunk, OutputChunk


def consume(outputs: list[OutputChunk]) -> None:
    for output in outputs:
        # Передавайте все блоки, если следующему этапу нужна полная история потока.
        archive_pcm(output.samples)
        if output.is_activated:
            command_recorder.write(output.samples)


samples = np.asarray(source_samples, dtype=np.dtype("<i2")).copy()
consume(
    processor.process(
        InputChunk(
            samples=samples,
            running_time_ns=source_running_time_ns,
            captured_at_ns=source_wall_time_ns,
            generation=capture_generation,
            discontinuity=continuity_was_lost,
        )
    )
)
```

`process()` заимствует память вызывающей стороны только на время вызова и копирует
весь блок до инференса. После возврата или исключения исходный массив можно
переиспользовать. Каждый `OutputChunk.samples` является независимым доступным для
записи массивом, принадлежащим получателю.

Метод возвращает готовый список, который может быть пустым. Размеры и границы
выходных блоков не являются частью контракта. Значения `is_speech` и `is_activated`
одинаковы для всех семплов блока и независимы друг от друга. Активированная тишина
является корректным результатом.

## Время и непрерывность

`running_time_ns` и `captured_at_ns` относятся к первому входному семплу. После
изменения границ выходное время первого семпла интерполируется:

```text
timestamp + floor(sample_offset * 1_000_000_000 / 16_000)
```

Если относительное время потока недоступно, используйте `running_time_ns=0`. Весь
непрерывный сегмент в этом случае остается без относительного времени. Нельзя
смешивать нулевое и ненулевое `running_time_ns` внутри одного сегмента.
`captured_at_ns=0` является обычной временной опорой, а не специальным значением.

Внутри сегмента временные опоры должны соответствовать количеству семплов с допуском
в один период, то есть 62 500 нс. Порядок всегда определяется позицией семпла, а не
временем.

Изменение `generation` или `discontinuity=True` создает жесткую границу. Кадры VAD и
ключевого слова ее не пересекают. Активация через такую границу сохраняется; если ее
тоже нужно сбросить, вызовите `rearm()`.

При интеграции с Devicelab поля `CapturedChunk` напрямую переносятся в `InputChunk`.
Колбэки Devicelab сериализованы и соответствуют требованиям процессора.

## Жизненный цикл

Всегда обрабатывайте значения, возвращенные lifecycle-методами:

```python
consume(processor.rearm())        # сбросить активацию, сохранить временную шкалу
consume(processor.discontinue()) # завершить сегмент, позже принять новый
consume(processor.finish())      # завершить конечный ввод
consume(processor.close())       # завершить ввод и закрыть оба бэкенда
```

Вызывайте `close()` в `finally`. `WakeStreamProcessor` не является контекстным
менеджером. После успешного выполнения `finish()` и `close()` терминальны для ввода;
их повторный вызов безопасен. `discontinue()` помечает следующий принятый блок как
разрыв, но не изменяет его поколение.

При границе или завершении неполные кадры моделей не дополняются искусственными
семплами. Реальный хвост все равно возвращается с консервативными метаданными.

## Ошибки обработки

Если после принятия аудио падает `process()`, `rearm()`, `discontinue()` или
финализация, возникает `StreamProcessingError`, а процессор становится терминально
ошибочным. Вызов с ошибкой не возвращает результат. Получите все принятое аудио
ровно один раз:

```python
from lumivox_wakelab import StreamCloseError, StreamProcessingError

try:
    consume(processor.process(chunk))
except StreamProcessingError:
    consume(processor.drain_failed())
    raise
finally:
    try:
        consume(processor.close())
    except StreamCloseError as error:
        consume(error.outputs)
        raise
```

`drain_failed()` не выполняет новый инференс. При необходимости `close()` сам
попытается выполнить такой слив. Если `close()` поднимает `StreamCloseError`,
обработайте `error.outputs`; в `error.errors` находятся все ошибки ресурсов.

## Многопоточность

Один процессор и два его бэкенда представляют один логический поток аудио. API
синхронный, нереентерабельный и не является thread-safe. Сериализуйте все вызовы
`process()`, lifecycle-методов, счетчиков и `drain_diagnostics()`. Не вызывайте один
процессор одновременно из потоков захвата, завершения и UI. Для независимых
аудиопотоков создавайте отдельные полностью инициализированные экземпляры.

Внутренних рабочих потоков и скрытых асинхронных очередей нет. Инференс в `process()`
выполняется в вызывающем потоке.
