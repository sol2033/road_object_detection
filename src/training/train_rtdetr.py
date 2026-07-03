from pathlib import Path
from ultralytics import RTDETR

DATA_YAML = Path("data/processed/yolo/data.yaml").resolve()
PROJECT_DIR = Path("results/rtdetr").resolve()


def train_rtdetr(epochs=30, imgsz=512, batch=2):
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        amp=False,
        workers=2,
        project=str(PROJECT_DIR),
        name="train",
        exist_ok=True,
    )