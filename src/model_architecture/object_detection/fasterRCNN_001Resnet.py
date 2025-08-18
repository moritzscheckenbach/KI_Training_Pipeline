import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


def get_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights, min_size=224, max_size=512)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=224, max_size=512)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---- Funktionen, die dein Loader erwartet ----


def build_model(num_classes: int):
    # "Fresh model" ohne Pretrained (wie in deinem try/else-Zweig)
    return get_model(num_classes=num_classes, pretrained=False)


def get_input_size():
    # Input Size für Faster R-CNN ist variabel, aber typischerweise 800x800
    return 224, 224
