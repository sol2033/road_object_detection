import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def count_matches(pred, gt, conf_threshold, iou_threshold):
    """Считает TP, FP, FN для 1 картинки.
    Предсказание = TP, если его класс совпал с реальным объектом
    и IoU с ним >= порога"""
    keep = pred["scores"] >= conf_threshold
    pred_boxes = pred["boxes"][keep]
    pred_labels = pred["labels"][keep]
    pred_scores = pred["scores"][keep]

    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0     
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)        

    # сначала обрабатываем самые уверенные предсказания
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    ious = box_iou(pred_boxes, gt_boxes)  

    matched_gt = set()
    tp = 0
    fp = 0
    for i in range(len(pred_boxes)):
        best_iou, best_j = ious[i].max(dim=0)
        best_iou = best_iou.item()
        best_j = best_j.item()
        same_class = pred_labels[i].item() == gt_labels[best_j].item()
        if best_iou >= iou_threshold and same_class and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


@torch.no_grad()
def evaluate(model, loader, device, conf_threshold=0.5, iou_threshold=0.5):
    """Считает mAP (torchmetrics) и Precision/Recall/F1 (вручную) на наборе."""
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        predictions = model(images)
        predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]

        metric.update(predictions, targets)  

        for pred, gt in zip(predictions, targets):
            tp, fp, fn = count_matches(pred, gt, conf_threshold, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    map_result = metric.compute()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    result = {
        "map": map_result["map"].item(),
        "map_50": map_result["map_50"].item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map_per_class": map_result["map_per_class"].tolist(),
        "classes": map_result["classes"].tolist(),
    }
    return result
