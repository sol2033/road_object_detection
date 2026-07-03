import argparse
import time
import traceback

import configs.default as cfg
from src.training.train_yolo import train_yolo
from src.training.train_rtdetr import train_rtdetr
from src.training.train_torchvision import train_model


def run_one(model_name):
    """Запускает обучение одной модели с настройками из конфига."""
    epochs = cfg.EPOCHS
    params = cfg.MODELS[model_name]

    if model_name == "yolo":
        train_yolo(epochs=epochs, imgsz=params["imgsz"], batch=params["batch"])
    elif model_name == "rtdetr":
        train_rtdetr(epochs=epochs, imgsz=params["imgsz"], batch=params["batch"])
    else:
        # torchvision-модели: fasterrcnn, retinanet, ssd
        train_model(model_name, epochs=epochs, batch_size=params["batch"], lr=params["lr"])


def run_all():
    """Обучает все модели по очереди.
    Если одна упадёт — остальные продолжат (try/except), в конце печатается сводка."""
    results = {}
    for name in cfg.MODELS:
        print(f"\n{'='*50}\nЗапуск обучения: {name}\n{'='*50}")
        start = time.time()
        try:
            run_one(name)
            elapsed = time.time() - start
            results[name] = f"OK ({elapsed/60:.1f} мин)"
        except Exception:
            results[name] = "ОШИБКА"
            traceback.print_exc()

    print(f"\n{'='*50}\nИТОГ\n{'='*50}")
    for name, status in results.items():
        print(f"{name}: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["yolo", "fasterrcnn", "retinanet", "ssd", "rtdetr", "all"])
    args = parser.parse_args()

    if args.model == "all":
        run_all()
    else:
        run_one(args.model)


if __name__ == "__main__":
    main()
