# Детекция объектов на KITTI

Учебная практика: детекция объектов на датасете KITTI (автономное вождение).
Обучаем и сравниваем 5 моделей на 4 классах (Car, Pedestrian, Cyclist, Truck):

- YOLO (yolo11n) — Ultralytics
- Faster R-CNN, RetinaNet, SSD — torchvision
- RT-DETR (rtdetr-l) — Ultralytics

## Структура

```
configs/default.py   — гиперпараметры (эпохи, batch, lr, пути)
data/                — датасет (raw — исходный KITTI, processed — сплиты и YOLO-формат)
src/dataset/         — разбиение данных и конвертация в нужные форматы
src/models/          — создание моделей (подмена головы под наши классы)
src/training/        — обучение моделей
src/evaluation/      — метрики (mAP, Precision, Recall, F1) и оценка на test
src/utils/           — визуализация (боксы, графики, предсказания)
notebooks/eda.ipynb  — анализ данных
results/             — веса, логи, метрики, графики
main.py              — точка входа для обучения
```

## Установка

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Датасет KITTI (2D Object Detection) положить в:
- `data/raw/training/image_2/` — картинки (.png)
- `data/raw/training/label_2/` — метки (.txt)

## Подготовка данных

```powershell
python -m src.dataset.split_dataset   # разбить на train/val/test (70/15/15)
python -m src.dataset.kitti_to_yolo   # сконвертировать в YOLO-формат
```

## Настройка гиперпараметров

Все гиперпараметры лежат в `configs/default.py`:

- `EPOCHS` — число эпох (одинаковое для всех моделей ради честного сравнения);
- `MODELS` — параметры под каждую модель: `imgsz` (размер картинки), `batch` (размер батча),
  `lr` (learning rate, для torchvision-моделей);
- `NUM_CLASSES`, `DATA_YAML`, `RESULTS_DIR`, `SEED` — число классов, пути и seed.

Чтобы поменять настройки обучения, правишь этот файл — `main.py` берёт значения оттуда.

## Обучение

Все скрипты запускаются из корня проекта как модули.

```powershell
python main.py --model yolo    # одна модель
python main.py --model all     # все 5 по очереди
```
Доступные модели: `yolo`, `fasterrcnn`, `retinanet`, `ssd`, `rtdetr`, `all`.
Веса и логи сохраняются в `results/`.

## Метрики и графики

```powershell
python -m src.evaluation.evaluate_test    # оценка всех моделей на test в results/test_metrics.csv
python -m src.utils.plots                 # графики сравнения и loss-кривые в results/plots/
python -m src.utils.visualize_predictions # предсказания на картинках в results/plots/predictions/
```

## Эксперименты

Перебор гиперпараметров на быстрых моделях:

```powershell
python -m src.training.experiment_yolo   # YOLO: перебор learning rate и размера картинки
python -m src.training.experiment_ssd    # SSD: обучение с аугментацией и сравнение с baseline
```
Результаты и графики экспериментов сохраняются в `results/`.
