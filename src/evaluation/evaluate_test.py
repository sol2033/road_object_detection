from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO, RTDETR

import configs.default as cfg
from src.models.detectors import get_model
from src.evaluation.metrics import evaluate
from src.training.train_torchvision import make_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_YAML = Path(cfg.DATA_YAML).resolve()


def eval_torchvision(model_name):
    """Грузит веса torchvision-модели и считает метрики на test."""
    weights = Path(cfg.RESULTS_DIR) / "torchvision" / model_name / "model.pth"
    model = get_model(model_name, cfg.NUM_CLASSES)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE)

    test_loader = make_loader("test", batch_size=4, shuffle=False)
    metrics = evaluate(model, test_loader, DEVICE)

    return {
        "mAP50": metrics["map_50"],
        "mAP50-95": metrics["map"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def eval_ultralytics(model_name):
    """Оценивает YOLO или RT-DETR на test встроенной оценкой Ultralytics."""
    weights = Path(cfg.RESULTS_DIR) / model_name / "train" / "weights" / "best.pt"
    if model_name == "yolo":
        model = YOLO(weights)
    else:
        model = RTDETR(weights)

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        project=str((Path(cfg.RESULTS_DIR) / model_name).resolve()),
        name="test_eval",
        exist_ok=True,
    )

    precision = results.box.mp   # средняя precision по классам
    recall = results.box.mr      # средний recall по классам
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    all_metrics = {}
    all_metrics["yolo"] = eval_ultralytics("yolo")
    for name in ["fasterrcnn", "retinanet", "ssd"]:
        all_metrics[name] = eval_torchvision(name)
    all_metrics["rtdetr"] = eval_ultralytics("rtdetr")

    # словарь словарей -> таблица: модели по строкам, метрики по столбцам
    df = pd.DataFrame.from_dict(all_metrics, orient="index").round(3)

    df.to_csv(Path(cfg.RESULTS_DIR) / "test_metrics.csv")
    print(df)


if __name__ == "__main__":
    main()
