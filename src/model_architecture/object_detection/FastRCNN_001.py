import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=224,
            max_size=512,
            # Modified parameters for Fast R-CNN behavior
            rpn_pre_nms_top_n_test=500,
            rpn_post_nms_top_n_test=300,
            box_batch_size_per_image=256,
        )
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=224, max_size=512, rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=300, box_batch_size_per_image=256)

    # Update the classification head to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---- Functions expected by your loader ----


def build_model(num_classes: int):
    # "Fresh model" without pretrained weights
    return get_model(num_classes=num_classes, pretrained=False)


def get_input_size():
    # Input size is variable, but we'll use the same as FasterRCNN
    return 224, 224


# Test function
if __name__ == "__main__":
    model = build_model(num_classes=20)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("FastRCNN_002 Model loaded successfully!")
    print(f"Input size: {get_input_size()}")
    print(f"Dummy output: {output}")
