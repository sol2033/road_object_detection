from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("data/processed/yolo/data.yaml").resolve()
PROJECT_DIR = Path("results/yolo").resolve()


def train_yolo(epochs=50, imgsz=640, batch=16, lr0=0.01, optimizer="auto", name="train"):
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer=optimizer,   # "auto" сам подбирает lr; для перебора lr нужен конкретный (SGD)
        project=str(PROJECT_DIR),
        name=name,
        exist_ok=True,
    )