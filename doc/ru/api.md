# Справочник API

Здесь описаны поддерживаемые импорты из `lumivox_wakelab`,
`lumivox_wakelab.vad` и `lumivox_wakelab.wakeword`. Имена из модулей с префиксом
подчеркивания являются внутренними.

## Потоковые значения

Все потоковые значения являются frozen dataclass со slots. Хранящиеся в блоках
NumPy-массивы при этом остаются доступными для записи.

### `InputChunk`

```python
InputChunk(samples, running_time_ns, captured_at_ns, generation, discontinuity)
```

| Поле | Значение |
| --- | --- |
| `samples` | Доступный для записи непустой одномерный `numpy.ndarray` с точным dtype `numpy.dtype("<i2")`. |
| `running_time_ns` | Неотрицательное относительное время первого семпла; ноль означает отсутствие времени для сегмента. |
| `captured_at_ns` | Неотрицательная опора настенных часов для первого семпла. |
| `generation` | Неотрицательное поколение источника. Изменение создает границу. |
| `discontinuity` | Потеряна ли непрерывность с предыдущим вводом. |

### `OutputChunk`

Содержит пять полей входных метаданных, а также `is_speech` и `is_activated`.
Два логических признака независимы. Метаданные едины для всех семплов блока. Первый
выход после входного разрыва, смены поколения или явного `discontinue()` имеет
`discontinuity=True`; последующие блоки этого сегмента имеют `False`.

## Настройка политик

### `VadPolicyConfig`

| Поле | Ограничение и назначение |
| --- | --- |
| `speech_threshold` | Конечная вероятность из `[0, 1]`; значение не ниже порога является свидетельством речи. |
| `silence_threshold` | Конечная вероятность из `[0, 1]`, строго ниже `speech_threshold`; значение ниже порога является свидетельством тишины. |
| `minimum_speech_samples` | Положительное число семплов до подтверждения речи. Округляется до полных кадров бэкенда. |
| `minimum_silence_samples` | Положительное число семплов до завершения речи. Округляется до полных кадров бэкенда. |
| `left_padding_samples` | Неотрицательный отступ тишины перед подтвержденной речью. |
| `right_padding_samples` | Неотрицательный отступ после первого свидетельства тишины. |

Вероятности между порогами сохраняют текущее состояние свидетельства. Значений по
умолчанию у политики нет.

### `WakePolicyConfig`

| Поле | Ограничение и назначение |
| --- | --- |
| `score_threshold` | Конечная вероятность из `[0, 1]`; значение не ниже порога считается срабатыванием. |
| `consecutive_score_count` | Положительное число последовательных срабатываний для активации. Промах обнуляет счетчик. |
| `silence_bridge_samples` | Неотрицательная тишина, допускаемая при завершении кандидата. |
| `pre_roll_samples` | Неотрицательное аудио до первого score подтверждающей серии, ограниченное началом кандидата. |

Если активированное аудио должно включать ключевое слово, `pre_roll_samples` обязан
покрывать измеренную задержку выбранного классификатора.

### `StreamConfig`

Объединяет `vad_policy`, `wake_policy` и положительное
`max_retained_audio_samples`. Для размера VAD-кадра `V` и wake-кадра `W` требуется:

```text
pre_roll + consecutive_score_count * W + silence_bridge
  + max(left_padding + ceil(minimum_speech / V) * V,
        ceil(minimum_silence / V) * V)
```

Граница относится к удерживаемому PCM с еще не принятыми решениями. В нее не входят
копия текущего блока вызывающей стороны и собираемые выходные массивы.

## `WakeStreamProcessor`

```python
WakeStreamProcessor(
    config,
    vad_backend,
    wake_backend,
    /,
    *,
    logger,
    diagnostic_event_capacity=0,
)
```

Логгер настраивается приложением. Нулевая емкость отключает диагностику,
положительная включает ограниченную очередь, удаляющую старые события при переполнении.

| Элемент | Контракт |
| --- | --- |
| `process(chunk)` | Только открытое состояние. Принимает весь блок и возвращает готовый результат. |
| `rearm()` | Сбрасывает оценку ключевого слова и активацию для будущего аудио, сохраняя timeline и VAD. |
| `discontinue()` | Завершает текущий сегмент и принудительно отмечает границу на следующем вводе. |
| `finish()` | Завершает все реальное аудио без padding кадров и навсегда заканчивает ввод. |
| `drain_failed()` | Только ошибочное состояние. Возвращает принятое аудио без нового инференса. |
| `drain_diagnostics()` | Возвращает и очищает диагностику в любом состоянии. |
| `close()` | Завершает или сливает поток, закрывает оба бэкенда и становится терминальным. |
| `accepted_samples` | Число семплов, принятых процессором за все время. |
| `emitted_samples` | Число семплов, возвращенных методами процессора. |
| `retained_samples` | Текущее число семплов с еще не принятым решением. |

`StreamProcessor` является структурным `Protocol` для семи методов. Счетчики
конкретного процессора в него не входят; runtime-проверка протокола не поддерживается.

## Ошибки потока

Все ошибки потока наследуют `StreamError`, который наследует `RuntimeError`.

| Ошибка | Значение |
| --- | --- |
| `StreamStateError` | Операция недопустима в текущем состоянии жизненного цикла. |
| `StreamProcessingError` | Обработка или финализация упала после возможного принятия блока. Используйте `drain_failed()`. |
| `StreamCloseError` | Ошибка финализации, слива или закрытия бэкендов. `outputs` содержит восстановленный результат, `errors` — все исходные исключения. |

Создание `InputChunk`, конфигурация и другие проверки до принятия данных поднимают
обычные `TypeError` или `ValueError`. Нарушения согласованности времени между блоками
обнаруживаются после принятия всего блока и поэтому дают `StreamProcessingError`.
Не повторяйте такой блок; восстановите его через `drain_failed()`.

## Протоколы бэкендов

`lumivox_wakelab.vad.VadBackend` и
`lumivox_wakelab.wakeword.WakeWordBackend` имеют одинаковый структурный контракт:

```python
class Backend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: NDArray[np.int16]) -> float: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

Частота должна быть 16 000 Гц, размер кадра должен быть положительным. `infer()`
получает ровно один нативный кадр и возвращает конечное значение из `[0, 1]`.
Операция транзакционна: исключение не меняет состояние потока. `reset()` возвращает
детерминированное начальное состояние, а `close()` терминален и идемпотентен.

## Готовые бэкенды

### `SileroVad`

Создавайте через `SileroVad.load(model_path, *, logger)`. Публичные свойства:
`sample_rate`, `frame_samples`, `model_version`, `active_providers`. Методы:
`infer()`, `reset()`, `close()`. Бэкенд поддерживает контекстный менеджер.

Ошибки: `VadError`, `VadInitializationError`, `VadInferenceError`, `VadClosedError`.
`load()` также может передать ошибку отсутствия или целостности артефакта, если общая
модель изменилась после получения.

### `OpenWakeWord`

```python
OpenWakeWord.load(
    feature_model_dir,
    classifier_path,
    *,
    logger,
    classifier_bytes=None,
)
```

При передаче `classifier_bytes` загружается этот неизменяемый снимок, а
`classifier_path` сохраняется как диагностическое имя. Это устраняет гонку между
проверкой дайджеста вызывающей стороной и повторным чтением файла.

Публичные свойства: `sample_rate`, `frame_samples`, `feature_version`,
`classifier_history_frames`, `active_providers`. Методы: `infer()`, `reset()`,
`close()`. Поддерживается контекстный менеджер.

Ошибки: `WakeWordError`, `WakeWordInitializationError`,
`WakeWordInferenceError`, `WakeWordClosedError`. `load()` также может передать ошибку
отсутствия или целостности артефакта, если общие feature-модели изменились после
получения.

## Артефакты и их ошибки

```python
ensure_silero_vad_model(
    model_root, *, logger, allow_download=True, timeout=30.0
) -> pathlib.Path

ensure_openwakeword_feature_models(
    model_root, *, logger, allow_download=True, timeout=30.0
) -> pathlib.Path
```

Обе функции проверяют фактические байты по размеру и SHA-256 из манифеста в коде.
Ошибки наследуют `ArtifactError`: `ArtifactMissingError`,
`ArtifactIntegrityError`, `ArtifactDownloadError`, `ArtifactPublicationError`.

## Диагностика

`drain_diagnostics()` возвращает `DiagnosticDrain(events, dropped_events)`. События
неизменяемы, содержат позиции семплов и скалярное состояние, но не PCM. При
переполнении удаляются старые события, а `dropped_events` растет до следующего слива.

Публичные типы событий:

- `VadFrameDiagnostic` и `VadSpeechDiagnostic`;
- `WakeCandidateStartedDiagnostic` и `WakeCandidateEndedDiagnostic`;
- `WakeFrameDiagnostic` и `ActivationDiagnostic`;
- `RearmDiagnostic` и `ContinuityBoundaryDiagnostic`;
- `FailureDiagnostic`.

Связанные `StrEnum`: `VadTransitionReason`, `WakeCandidateEndReason`,
`ContinuityBoundaryReason`, `DiagnosticFailureOperation`. Диапазоны семплов являются
полуинтервалами `[start_sample, end_sample)` на общей шкале процессора.
