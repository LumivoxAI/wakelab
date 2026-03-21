# Модели и артефакты

## Разделение ответственности

Wakelab поставляет адаптеры инференса, но не файлы моделей. Для общих upstream-
артефактов в коде зафиксированы манифесты. Классификатор ключевого слова остается
доверенным ресурсом вызывающей стороны. Получение файлов, проверка, создание ONNX-
сессий и прогрев явно выполняются до потоковой обработки.

## Silero VAD

VAD-адаптер использует стандартную ONNX-модель Silero VAD v6.2.1:

- путь кеша: `<model-root>/silero-vad/v6.2.1/silero_vad.onnx`;
- нативный кадр: 512 семплов, или 32 мс при 16 кГц;
- provider: ONNX Runtime CPU;
- SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

```python
from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model

model_path = ensure_silero_vad_model("model-data", logger=logger)
vad = SileroVad.load(model_path, logger=logger)
```

`load()` проверяет граф и provider, прогревает инференс и сбрасывает рекуррентное
состояние. `infer()` возвращает только вероятность речи; пороги и временная политика
настраиваются в потоковом процессоре.

## openWakeWord

Wake-бэкенд использует feature ABI mel-spectrogram и embedding из openWakeWord
v0.5.1:

- каталог кеша: `<model-root>/openwakeword-features/v0.5.1/`;
- общие файлы: `melspectrogram.onnx` и `embedding_model.onnx`;
- нативный кадр: 1 280 семплов, или 80 мс при 16 кГц;
- provider: ONNX Runtime CPU.

```python
from lumivox_wakelab.wakeword import OpenWakeWord, ensure_openwakeword_feature_models

feature_dir = ensure_openwakeword_feature_models("model-data", logger=logger)
wake = OpenWakeWord.load(feature_dir, "models/my-wake-word.onnx", logger=logger)
```

Классификатор вызывающей стороны должен быть бинарной ONNX-моделью с одним входом
float32 формы `[1, N, 96]`, где `1 <= N <= 120`, и одним выходом float32 формы
`[1, 1]`. Wakelab проверяет этот тензорный контракт, но не устанавливает
происхождение, целостность, качество, порог, временную калибровку или лицензию.

Чтобы безопасно проверить SHA-256, прочитайте и проверьте файл один раз, а затем
передайте те же байты в `load()`:

```python
import hashlib

classifier_path = "models/my-wake-word.onnx"
classifier_bytes = open(classifier_path, "rb").read()
actual = hashlib.sha256(classifier_bytes).hexdigest()
if actual != expected_sha256:
    raise RuntimeError("wake-word classifier digest mismatch")

wake = OpenWakeWord.load(
    feature_dir,
    classifier_path,
    logger=logger,
    classifier_bytes=classifier_bytes,
)
```

`load()` делает снимки всех моделей, проверяет схемы и provider, вычисляет
детерминированную историю тишины, прогревает полный конвейер и сбрасывает его.
`reset()` восстанавливает эту историю без ONNX-инференса.

## Целостность кеша

Функции `ensure_*` при каждом вызове проверяют реальные байты по зафиксированным
размеру и SHA-256. Соседний `.sha256` является метаданными кеша, а не источником
доверия. Загрузка записывается во временный файл, проверяется, синхронизируется и
атомарно публикуется. Частичный или некорректный файл не заменяет корректный.

На Linux получение артефакта использует межпроцессную блокировку. Выполняется одна
ограниченная попытка; расписание повторов определяет приложение.

После сетевой подготовки включите офлайн-режим явно:

```python
vad_path = ensure_silero_vad_model(
    model_root,
    logger=logger,
    allow_download=False,
)
feature_dir = ensure_openwakeword_feature_models(
    model_root,
    logger=logger,
    allow_download=False,
)
```

Офлайн-режим запрещает загрузки. Восстановление метаданных кеша и lock-файлы все
равно могут записываться в каталог моделей.

## Лицензии

Закрепленная модель Silero распространяется авторами под MIT. Узкая реализация
инференса openWakeWord использует Apache-2.0. Готовые upstream-классификаторы
openWakeWord имеют лицензию CC BY-NC-SA 4.0 и поэтому намеренно не загружаются и не
распространяются Wakelab. Изучите [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
и лицензию каждого классификатора приложения.
