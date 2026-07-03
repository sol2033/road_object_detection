from pathlib import Path
from PIL import Image
import shutil

# Соответствие имени класса KITTI и индекса YOLO.
CLASS_TO_ID = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Truck": 3,
}

# Исходные данные KITTI
IMAGES_DIR = Path("data/raw/training/image_2")
LABELS_DIR = Path("data/raw/training/label_2")
SPLITS_DIR = Path("data/processed/splits")

# Где будет результат для yolo
YOLO_DIR = Path("data/processed/yolo")

def convert_label(label_path, img_width, img_height):
    """Конвертирует одну метку KITTI в список строк YOLO формата.
    Объекты не из CLASS_TO_ID пропускаем, таким образом фильтруя классы."""
    yolo_lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            cls = parts[0]

            # фильтрация только нужных классов
            if cls not in CLASS_TO_ID:
                continue

            class_id = CLASS_TO_ID[cls]

            # бокс KITTI
            x1, y1, x2, y2 = map(float, parts[4:8])

            # переводим из kitti координат в yolo
            box_w = x2 - x1
            box_h = y2 - y1
            cx = x1 + box_w / 2
            cy = y1 + box_h / 2

            # нормализуем в доли относительно размера картинки
            cx /= img_width
            cy /= img_height
            box_w /= img_width
            box_h /= img_height

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")

    return yolo_lines


def read_split(split_name):
    """Читает список id картинок из файла разбиения (train и прочее)."""
    split_file = SPLITS_DIR / f"{split_name}.txt"
    return split_file.read_text().splitlines()


def convert_split(split_name):
    """Конвертирует один сплит. копирует картинки и
    создаёт YOLO-метки в нужной структуре папок."""
    image_ids = read_split(split_name)

    # папки назначения: images/train, labels/train и т.д.
    img_out_dir = YOLO_DIR / "images" / split_name
    lbl_out_dir = YOLO_DIR / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for image_id in image_ids:
        src_img = IMAGES_DIR / f"{image_id}.png"
        src_lbl = LABELS_DIR / f"{image_id}.txt"

        # читаем размер картинки. Это надо для нормализации
        with Image.open(src_img) as im:
            img_w, img_h = im.size

        yolo_lines = convert_label(src_lbl, img_w, img_h)

        # пропуск картинок, где нет ни одного нужного класса
        if not yolo_lines:
            continue

        # копируем картинку в YOLO-папку
        shutil.copy(src_img, img_out_dir / f"{image_id}.png")
        # пишем YOLO-метку
        (lbl_out_dir / f"{image_id}.txt").write_text("\n".join(yolo_lines))
        kept += 1

    print(f"{split_name}: {kept} картинок с объектами -> {img_out_dir}")


def write_data_yaml():
    """Создаёт data.yaml — конфиг датасета для Ultralytics."""
    # классы по порядку id
    names = sorted(CLASS_TO_ID, key=CLASS_TO_ID.get) 
    lines = [
        f"path: {YOLO_DIR.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(names)}",
        f"names: {names}",
    ]
    (YOLO_DIR / "data.yaml").write_text("\n".join(lines))
    print(f"data.yaml создан -> {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
    write_data_yaml()