import json
from pathlib import Path
import configs.default as cfg
from src.training.train_torchvision import train_model
from src.utils.plots import plot_ssd_experiment, PLOTS_DIR

RESULTS_DIR = Path(cfg.RESULTS_DIR)

# столько же эпох, сколько у baseline SSD — иначе сравнение нечестное
EPOCHS = 30


def load_metrics(run_name):
    """Читает metrics.json конкретного прогона torchvision."""
    path = RESULTS_DIR / "torchvision" / run_name / "metrics.json"
    return json.loads(path.read_text())


def main():
    ssd = cfg.MODELS["ssd"]

    # обучаем SSD С аугментацией в отдельную папку ssd_aug (baseline без неё не трогаем)
    train_model("ssd", epochs=EPOCHS, batch_size=ssd["batch"], lr=ssd["lr"],
                augment=True, run_name="ssd_aug")

    no_aug = load_metrics("ssd")       # baseline (без аугментации)
    aug = load_metrics("ssd_aug")      # с аугментацией

    print("\n=== SSD: аугментация off -> on ===")
    for key in ["map_50", "map", "precision", "recall", "f1"]:
        print(f"{key:>10}: {no_aug[key]:.3f} -> {aug[key]:.3f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_ssd_experiment(no_aug, aug)
    print("График -> results/plots/ssd_experiment.png")


if __name__ == "__main__":
    main()
