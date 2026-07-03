from pathlib import Path
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt

import configs.default as cfg

RESULTS_DIR = Path(cfg.RESULTS_DIR)
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_model_comparison():
    """Столбчатая диаграмма mAP по всем моделям (из test_metrics.csv)."""
    df = pd.read_csv(RESULTS_DIR / "test_metrics.csv", index_col=0)

    models = list(df.index)
    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    # два набора столбцов рядом: сдвигаем их влево/вправо на половину ширины
    plt.bar([i - width / 2 for i in x], df["mAP50"], width, label="mAP@50")
    plt.bar([i + width / 2 for i in x], df["mAP50-95"], width, label="mAP@50-95")
    plt.xticks(list(x), models)
    plt.ylabel("mAP")
    plt.title("Сравнение моделей на test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png")
    plt.close()


def plot_loss_curves():
    """Loss-кривые torchvision-моделей, каждая в своём подграфике."""
    names = ["fasterrcnn", "retinanet", "ssd"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, name in zip(axes, names):
        loss_file = RESULTS_DIR / "torchvision" / name / "loss_history.json"
        losses = json.loads(loss_file.read_text())
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o")
        ax.set_title(name)
        ax.set_xlabel("эпоха")
        ax.set_ylabel("средний loss")

    fig.suptitle("Loss-кривые обучения (torchvision)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "loss_curves.png")
    plt.close()


def read_yolo_map50(run_name):
    """mAP@50 последней эпохи из results.csv прогона YOLO."""
    df = pd.read_csv(RESULTS_DIR / "yolo" / run_name / "results.csv")
    df.columns = df.columns.str.strip()   # у Ultralytics бывают пробелы в именах столбцов
    return df["metrics/mAP50(B)"].iloc[-1]


def plot_yolo_experiment(lr_values, imgsz_values):
    """Два графика: mAP@50 vs learning rate и mAP@50 vs размер картинки."""
    lr_maps = [read_yolo_map50(f"exp_lr_{lr}") for lr in lr_values]
    imgsz_maps = [read_yolo_map50(f"exp_imgsz_{s}") for s in imgsz_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lr_values, lr_maps, marker="o")
    ax1.set_xscale("log")   # lr удобнее на логарифмической оси
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("mAP@50 (val)")
    ax1.set_title("Влияние learning rate")

    ax2.plot(imgsz_values, imgsz_maps, marker="o")
    ax2.set_xlabel("размер картинки (imgsz)")
    ax2.set_ylabel("mAP@50 (val)")
    ax2.set_title("Влияние разрешения")

    fig.suptitle("YOLO: зависимость качества от гиперпараметров")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "yolo_experiments.png")
    plt.close()


def plot_ssd_experiment(no_aug, aug):
    """Столбцы: метрики SSD без и с аугментацией."""
    keys = ["map_50", "map", "precision", "recall", "f1"]
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall", "F1"]
    no_vals = [no_aug[k] for k in keys]
    aug_vals = [aug[k] for k in keys]

    x = range(len(keys))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], no_vals, width, label="без аугментации")
    plt.bar([i + width / 2 for i in x], aug_vals, width, label="с аугментацией")
    plt.xticks(list(x), labels)
    plt.ylabel("значение")
    plt.title("SSD: влияние аугментации")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ssd_experiment.png")
    plt.close()


def collect_ultralytics_plots():
    """Копирует готовые графики Ultralytics (YOLO, RT-DETR) в общую папку plots/."""
    for model in ["yolo", "rtdetr"]:
        for subdir in ["train", "test_eval"]:
            src_dir = RESULTS_DIR / model / subdir
            if not src_dir.exists():
                continue
            # берём все png из папки и копируем с префиксом, чтобы не перезаписать
            for png in src_dir.glob("*.png"):
                dst = PLOTS_DIR / f"{model}_{subdir}_{png.name}"
                shutil.copy(png, dst)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison()
    plot_loss_curves()
    collect_ultralytics_plots()
    print(f"Графики сохранены в {PLOTS_DIR}")


if __name__ == "__main__":
    main()
