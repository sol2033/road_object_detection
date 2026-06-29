import random
from pathlib import Path


RANDOM_SEED = 42

# На тесты в итоге останется 15% от датасета
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


# Пути относительно корня проекта
IMAGES_DIR = Path("data/raw/training/image_2")
SPLITS_DIR = Path("data/processed/splits")


def split_dataset():
    # собираем имена всех картинок без расширения
    image_ids = [p.stem for p in IMAGES_DIR.glob("*.png")]
    image_ids.sort() 

    random.seed(RANDOM_SEED)
    random.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_file = SPLITS_DIR / f"{name}.txt"
        split_file.write_text("\n".join(ids))
        print(f"{name}: {len(ids)} картинок -> {split_file}")


if __name__ == "__main__":
    split_dataset()