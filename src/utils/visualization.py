import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes(image, boxes, labels, color="red", save_path=None):
    """Рисует боксы и подписи классов поверх картинки.
    image  — объект PIL.Image (или массив для imshow);
    boxes  — список [x1, y1, x2, y2] в пикселях;
    labels — список подписей (строк) той же длины, что boxes;
    save_path — если задан, картинка сохраняется в файл, иначе показывается."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color=color, fontsize=9)

    ax.axis("off")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
