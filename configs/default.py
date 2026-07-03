# Воспроизводимость
SEED = 42

# Данные
NUM_CLASSES = 5                                # 4 класса + фон (для torchvision)
DATA_YAML = "data/processed/yolo/data.yaml"    # конфиг датасета для YOLO/RT-DETR

# Куда складывать результаты
RESULTS_DIR = "results"

# Единое число эпох. одинаковое у всех моделей ради честного сравнения
EPOCHS = 30

# Параметры под каждую модель. Разные, т.к. есть ограничение по ресурсам (GPU RTX 4060 8GB)
MODELS = {
    "yolo":       {"imgsz": 640, "batch": 16},
    "fasterrcnn": {"batch": 4, "lr": 0.005},
    "retinanet":  {"batch": 4, "lr": 0.001},  
    "ssd":        {"batch": 8, "lr": 0.001},   
    "rtdetr":     {"imgsz": 512, "batch": 2},
}
