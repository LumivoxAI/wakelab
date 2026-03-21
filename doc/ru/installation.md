# Установка

## Поддерживаемая среда

Wakelab 0.1 поддерживает CPython 3.13 и 3.14 на Linux. Формат публичного аудио:
один канал, 16 000 Гц, знаковый 16-битный little-endian PCM. Сейчас ONNX работает
только через CPU provider.

Основные зависимости:

- `lumivox-core` из Git-репозитория;
- `numpy>=2,<3`;
- `onnxruntime>=1.24.1,<2`.

Захват микрофона через Devicelab и стенд NiceGUI входят только в зависимости для
разработки репозитория, но не в зависимости устанавливаемого пакета Wakelab.

## Установка из Git

Сейчас проект подключается из ветки `master`. Для `uv`:

```console
uv add "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

Для `pip`:

```console
python -m pip install "lumivox-wakelab @ git+https://github.com/LumivoxAI/wakelab.git@master"
```

Для воспроизводимого развертывания замените `master` на проверенный SHA коммита и
зафиксируйте полученный lock-файл.

## Репозиторий для разработки

Проект использует [uv](https://docs.astral.sh/uv/) и
[just](https://just.systems/):

```console
git clone https://github.com/LumivoxAI/wakelab.git
cd wakelab
just postclone
```

`just postclone` создает локальную среду и устанавливает все группы зависимостей.
После обычных изменений зависимостей используйте `just sync`.

## Требования к моделям

В Python-пакете нет ONNX-файлов. До создания процессора необходимо:

1. Получить закрепленную модель Silero VAD через `ensure_silero_vad_model()`.
2. Получить закрепленные feature-модели openWakeWord через
   `ensure_openwakeword_feature_models()`.
3. Предоставить локальный доверенный бинарный ONNX-классификатор, совместимый с
   feature ABI openWakeWord v0.5.1.

Первые две операции могут обращаться к сети. Их нужно выполнять на этапе настройки,
а не внутри аудиоколбэка. Подробнее: [Модели и артефакты](models.md).

## Проверка установки

```console
uv run python -c "import lumivox_wakelab; print('Wakelab import succeeded')"
```

Сам импорт пакета не загружает и не инициализирует модели.
