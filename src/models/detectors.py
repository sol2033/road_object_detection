import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
import torch.nn as nn


def get_model(model_name, num_classes):
    """Загружает предобученный детектор и подменяет голову под наши классы."""
    if model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes,
            norm_layer=partial(nn.GroupNorm, 32),
        )

    elif model_name == "ssd":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        # достаём параметры существующей головы
        in_channels = torchvision.models.detection._utils.retrieve_out_channels(
            model.backbone, (320, 320)
        )
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer
        )

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model
