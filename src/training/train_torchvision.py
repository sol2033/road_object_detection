import torch
from torch.utils.data import DataLoader
from src.dataset.kitti_torch_dataset import KittiDataset
from src.models.detectors import get_model
from src.evaluation.metrics import evaluate
import json
from pathlib import Path

# Число классов
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path("results/torchvision")


def train_model(model_name, epochs=10, batch_size=4, lr=0.005, augment=False, run_name=None):
    """Полный цикл обучения одной torchvision-модели с сохранением."""
    if run_name is None:
        run_name = model_name

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    train_loader = make_loader("train", batch_size=batch_size, shuffle=True, augment=augment)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # папка под результаты этого прогона
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        loss_history.append(avg_loss)

    # сохранение весос модели и историю loss
    torch.save(model.state_dict(), out_dir / "model.pth")
    (out_dir / "loss_history.json").write_text(json.dumps(loss_history))

    val_loader = make_loader("val", batch_size=batch_size, shuffle=False)
    metrics = evaluate(model, val_loader, DEVICE)
    # переводим тензоры в обычные числа для возможности сохранить в json
    metrics_clean = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_clean, indent=2))

    print(f"Веса, loss и метрики сохранены в {out_dir}")
    print(f"mAP@50: {metrics['map_50']:.4f}, mAP@50-95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    return model


def train_one_epoch(model, loader, optimizer, epoch):
    """Обучает модель один проход по всем батчам (одна эпоха)."""
    model.train() 
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Эпоха {epoch}: средний loss = {avg_loss:.4f}")
    return avg_loss

def collate_fn(batch):
    """Собирает батч из пар (картинка, target).
    Картинки и target остаются списками, т.к. у них разный размер
    и разное число объектов, поэтому стандартное складывание не подходит."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def make_loader(split, batch_size=4, shuffle=False, augment=False):
    """Создаёт DataLoader для нужного сплита."""
    dataset = KittiDataset(split, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return loader