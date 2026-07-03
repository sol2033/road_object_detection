from pathlib import Path
import torch
import torchvision.transforms.functional as F
from PIL import Image
from ultralytics import RTDETR

import configs.default as cfg
from src.models.detectors import get_model
from src.utils.visualization import draw_boxes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# у torchvision классы с 1 (0 = фон) — переводим номер обратно в название
ID_TO_CLASS = {1: "Car", 2: "Pedestrian", 3: "Cyclist", 4: "Truck"}

IMAGES_DIR = Path("data/raw/training/image_2")
SPLITS_DIR = Path("data/processed/splits")
OUT_DIR = Path(cfg.RESULTS_DIR) / "plots" / "predictions"


def visualize_fasterrcnn(num_images=4, conf_threshold=0.5):
    """Прогоняет Faster R-CNN на нескольких test-картинках и рисует предсказания."""
    model = get_model("fasterrcnn", cfg.NUM_CLASSES)
    weights = Path(cfg.RESULTS_DIR) / "torchvision" / "fasterrcnn" / "model.pth"
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_ids = (SPLITS_DIR / "test.txt").read_text().splitlines()[:num_images]

    for image_id in test_ids:
        image = Image.open(IMAGES_DIR / f"{image_id}.png").convert("RGB")
        img_tensor = F.to_tensor(image).to(DEVICE)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        # оставляем уверенные предсказания и готовим подписи вида "Car 0.92"
        boxes = []
        labels = []
        for box, label_id, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < conf_threshold:
                continue
            boxes.append(box.tolist())
            name = ID_TO_CLASS[label_id.item()]
            labels.append(f"{name} {score.item():.2f}")

        draw_boxes(image, boxes, labels, save_path=OUT_DIR / f"fasterrcnn_{image_id}.png")


def visualize_rtdetr(num_images=4, conf_threshold=0.5):
    """RT-DETR: Ultralytics сама рисует боксы+уверенность и сохраняет картинки."""
    weights = Path(cfg.RESULTS_DIR) / "rtdetr" / "train" / "weights" / "best.pt"
    model = RTDETR(weights)

    test_ids = (SPLITS_DIR / "test.txt").read_text().splitlines()[:num_images]
    image_paths = [str(IMAGES_DIR / f"{i}.png") for i in test_ids]

    model.predict(
        image_paths,
        conf=conf_threshold,
        save=True,
        project=str(OUT_DIR.resolve()),
        name="rtdetr",
        exist_ok=True,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    visualize_fasterrcnn()
    visualize_rtdetr()
    print(f"Картинки с предсказаниями сохранены в {OUT_DIR}")


if __name__ == "__main__":
    main()
