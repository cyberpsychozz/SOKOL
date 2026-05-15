# SOKOL

SOKOL — это пайплайн для детекции дыма и огня на изображениях и видео с дрона. Репозиторий покрывает полный цикл: подготовку датасета, обучение модели, экспорт в `ONNX` и инференс на локальной машине или на бортовом устройстве, например Orange Pi 5.

## Что умеет проект

- конвертирует `FASDD_UAV` в стандартную структуру датасета `YOLO`
- обучает детектор `RT-DETR-L` на классах `fire` и `smoke`
- экспортирует обученные веса в `ONNX`
- запускает инференс на видео, камере и edge-устройствах
- применяет лёгкий препроцессинг кадров для сложных aerial-сцен

## Текущее состояние

Сейчас основной тренировочный пайплайн построен вокруг `RT-DETR-L` из Ultralytics.

Метрики валидации на 34-й эпохе:

- `precision`: `0.882`
- `recall`: `0.876`
- `mAP50`: `0.914`
- `mAP50-95`: `0.652`

Для этой задачи основной метрикой качества лучше считать `mAP50-95`, а с прикладной точки зрения особенно важно следить за `recall`, потому что пропуск дыма или огня обычно критичнее, чем лишнее срабатывание.

## Структура репозитория

```text
SOKOL/
├── dataset_prepare.py              # Конвертация FASDD_UAV в YOLO-структуру
├── requirements.txt
├── training/
│   ├── train_rtdetr_l.py          # Обучение RT-DETR-L
│   ├── export_onnx.py             # Экспорт .pt чекпоинта в ONNX
│   ├── sokol_dataset.yaml         # Шаблон конфигурации датасета
│   └── README.md
└── MVP/
    └── src/
        ├── preprocessing.py       # Препроцессинг кадров с дрона
        ├── video_tracker.py       # Локальный инференс на видео
        ├── orangepi_inference.py  # Инференс для Orange Pi 5
        └── FastApi_Image_test.py  # Простой HTTP API для тестов
```

## Установка

Склонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/cyberpsychozz/SOKOL.git
cd SOKOL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если вы планируете обучение на GPU, установите сборку PyTorch, совместимую с вашей версией CUDA и архитектурой видеокарты.

## 1. Подготовка датасета

В репозитории есть конвертер для `FASDD_UAV`, который использует аннотации из `annotations/YOLO_UAV`.

Пример запуска:

```bash
python3 dataset_prepare.py convert-fasdd-uav \
  --source-root /path/to/FASDD_UAV \
  --output-dir datasets/fasdd_uav_yolo \
  --class-names fire smoke
```

Если не хочется копировать все файлы во время локальных экспериментов, можно использовать симлинки:

```bash
python3 dataset_prepare.py convert-fasdd-uav \
  --source-root /path/to/FASDD_UAV \
  --output-dir datasets/fasdd_uav_yolo \
  --class-names fire smoke \
  --symlinks
```

После конвертации будет создана такая структура:

```text
datasets/fasdd_uav_yolo/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
  dataset.yaml
```

## 2. Обучение RT-DETR-L

Базовая команда запуска:

```bash
python3 training/train_rtdetr_l.py \
  --data datasets/fasdd_uav_yolo/dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0
```

Полезные замечания:

- на более слабых GPU лучше уменьшить `--batch` до `2` или `4`
- если сборка CUDA/PyTorch не совместима с вашей видеокартой, скрипт теперь завершится раньше с более понятной ошибкой
- лучший чекпоинт обычно сохраняется в `training/runs/sokol_rtdetr_l/weights/best.pt`

## 3. Экспорт в ONNX

После обучения модель можно экспортировать в `ONNX` для дальнейшего edge-инференса:

```bash
python3 training/export_onnx.py \
  --weights training/runs/sokol_rtdetr_l/weights/best.pt \
  --imgsz 640 \
  --device cpu \
  --simplify
```

Зачем нужен `ONNX`:

- его проще использовать на edge-устройствах
- он переносимее, чем тренировочный `.pt` чекпоинт
- это удобная промежуточная точка для будущей конвертации в board-specific runtime

## 4. Локальный инференс на видео

`MVP/src/video_tracker.py` — это локальный пример обработки видео с препроцессингом и детекцией объектов.

Скрипт сейчас:

- загружает модель из `MVP/models`
- загружает видео из `MVP/data`
- обрабатывает кадры через `DroneFramePreprocessor`
- запускает детекцию и сохраняет результат в `MVP/results`

Поддерживаемые типы моделей:

- `YOLO .pt`
- `RT-DETR .pt`
- экспортированный `ONNX`

Важный момент:

- если ваш чекпоинт `RT-DETR` называется просто `best.pt`, автоопределение может принять его за `YOLO`
- в этом случае лучше переименовать файл, например в `best_rtdetr.pt`, или явно указать тип модели в коде

Пример запуска:

```bash
python3 MVP/src/video_tracker.py
```

## 5. Инференс на Orange Pi 5

`MVP/src/orangepi_inference.py` — основной reference-скрипт для запуска модели на Orange Pi 5.

Он поддерживает:

- камеру по индексу, например `0`
- путь к видеофайлу
- RTSP-поток
- загрузку `YOLO`, `RT-DETR` и `ONNX`
- пропуск кадров через `--frame-stride`
- CPU-first сценарий запуска

Рекомендуемая стартовая команда:

```bash
python3 MVP/src/orangepi_inference.py \
  --model /path/to/best.onnx \
  --source 0 \
  --device cpu \
  --imgsz 512 \
  --frame-stride 2 \
  --output MVP/results/orangepi_demo.mp4
```

Если тип модели определяется неоднозначно, его можно задать вручную:

```bash
python3 MVP/src/orangepi_inference.py \
  --model /path/to/best.pt \
  --model-type rtdetr \
  --source 0 \
  --device cpu
```

## 6. Пайплайн препроцессинга

В `MVP/src/preprocessing.py` сейчас реализован более практичный препроцессинг для кадров с дрона.

Доступные этапы:

- temporal smoothing
- `CLAHE` для локального усиления контраста
- опциональный bilateral denoising
- gamma correction
- мягкий sharpen
- опциональный gray-world white balance

Текущий профиль намеренно сделан достаточно аккуратным: он улучшает видимость дыма и деталей сцены, но не должен сильно портить текстуру дыма или слишком нагружать CPU на edge-устройстве.

## 7. FastAPI-демо

`MVP/src/FastApi_Image_test.py` — это простой HTTP API для тестов инференса по отдельным изображениям. Он удобен для быстрых ручных проверок, но не является основным сценарием deployment для Orange Pi.

## Рекомендуемый рабочий сценарий

1. Сконвертировать `FASDD_UAV` в YOLO-формат через `dataset_prepare.py`.
2. Обучить `RT-DETR-L` через `training/train_rtdetr_l.py`.
3. Экспортировать лучший чекпоинт через `training/export_onnx.py`.
4. Проверить инференс локально через `MVP/src/video_tracker.py`.
5. Перенести экспортированную модель на Orange Pi 5 и запустить `MVP/src/orangepi_inference.py`.

## Что можно улучшить дальше

- добавить per-class отчёты по `fire` и `smoke`
- отдельно замерить производительность `ONNX Runtime` без обёртки Ultralytics
- добавить прямой экспорт или пайплайн под `RKNN`
- перевести `video_tracker.py` на CLI-аргументы вместо констант в коде

## Лицензия

При необходимости добавьте сюда выбранную лицензию проекта перед публикацией или внешним распространением.
