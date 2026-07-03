from pathlib import Path
import configs.default as cfg
from src.training.train_yolo import train_yolo
from src.utils.plots import plot_yolo_experiment, read_yolo_map50, PLOTS_DIR

RESULTS_DIR = Path(cfg.RESULTS_DIR)

# что перебираем на каждом этапе
LR_VALUES = [0.001, 0.01, 0.1]
IMGSZ_VALUES = [320, 512, 640]


def sweep_lr(imgsz, batch):
    """Эксперимент 1: перебор learning rate (optimizer=SGD, иначе auto игнорирует lr0)."""
    for lr in LR_VALUES:
        print(f"\n=== lr={lr} ===")
        train_yolo(epochs=cfg.EPOCHS, imgsz=imgsz, batch=batch,
                   lr0=lr, optimizer="SGD", name=f"exp_lr_{lr}")

    # выбираем lr с максимальным mAP@50
    scores = {lr: read_yolo_map50(f"exp_lr_{lr}") for lr in LR_VALUES}
    best_lr = max(scores, key=scores.get)
    print(f"Лучший lr: {best_lr} (mAP@50={scores[best_lr]:.3f})")
    return best_lr


def sweep_imgsz(batch):
    """Эксперимент 2: перебор размера картинки (уже обученные пропускаем)."""
    for imgsz in IMGSZ_VALUES:
        run = f"exp_imgsz_{imgsz}"
        if (RESULTS_DIR / "yolo" / run / "results.csv").exists():
            print(f"imgsz={imgsz} уже обучен, пропускаю")
            continue
        print(f"\n=== imgsz={imgsz} ===")
        train_yolo(epochs=cfg.EPOCHS, imgsz=imgsz, batch=batch, name=run)


def main():
    batch = cfg.MODELS["yolo"]["batch"]
    base_imgsz = cfg.MODELS["yolo"]["imgsz"]

    sweep_lr(base_imgsz, batch)
    sweep_imgsz(batch)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_yolo_experiment(LR_VALUES, IMGSZ_VALUES)
    print("Эксперименты завершены, график -> results/plots/yolo_experiments.png")


if __name__ == "__main__":
    main()
