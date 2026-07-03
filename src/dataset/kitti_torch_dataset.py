from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

# Классы с индексами 
CLASS_TO_ID = {
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
    "Truck": 4,
}

IMAGES_DIR = Path("data/raw/training/image_2")
LABELS_DIR = Path("data/raw/training/label_2")
SPLITS_DIR = Path("data/processed/splits")


class KittiDataset(Dataset):
    """Читает KITTI и отдаёт картинки + боксы в формате torchvision."""

    def __init__(self, split, augment=False):
        split_file = SPLITS_DIR / f"{split}.txt"
        self.image_ids = split_file.read_text().splitlines()
        # аугментации включаем только для train (для val/test нельзя — оцениваем на честных данных)
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = IMAGES_DIR / f"{image_id}.png"
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        label_path = LABELS_DIR / f"{image_id}.txt"
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                cls = parts[0]
                if cls not in CLASS_TO_ID:
                    continue
                x1, y1, x2, y2 = map(float, parts[4:8])
                # Пропуск битых боксов (в KITTI такого нет, но подстраховка)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_ID[cls])

        if self.augment:
            image, boxes = self.apply_augment(image, boxes)

        image = F.to_tensor(image)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return image, target

    def apply_augment(self, image, boxes):
        """Аугментации для train: цветовые искажения + горизонтальный флип."""
        # color jitter меняет только пиксели, боксы не трогаем
        image = self.color_jitter(image)

        # горизонтальный флип с вероятностью 0.5
        if random.random() < 0.5:
            width = image.width
            image = F.hflip(image)
            # зеркалим x-координаты: левый и правый края меняются местами
            flipped = []
            for x1, y1, x2, y2 in boxes:
                flipped.append([width - x2, y1, width - x1, y2])
            boxes = flipped

        return image, boxes
    
