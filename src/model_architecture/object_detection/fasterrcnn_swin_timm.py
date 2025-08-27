# model_architecture/object_detection/fasterrcnn_swin_timm.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign


class TimmSwinBackbone(nn.Module):
    """
    Swin backbone from timm + FPN for Faster R-CNN.
    Variant B: img_size is fixed at (640, 640) during creation.
    => Inputs to the model MUST be 640x640 (otherwise AssertionError in PatchEmbed).

    Returns: Dict[str, Tensor] with keys "0","1","2","3" (compatible with ROIAlign).
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",  # timm ID that exists
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        fpn_out_channels: int = 256,
        img_size: Tuple[int, int] = (640, 640),  # <- fixed target size
    ):
        super().__init__()
        # timm Feature-Extraktor mit fester img_size
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,  # IMPORTANT: enforces PatchEmbed assertion for 640x640
        )

        # Channel numbers per stage (e.g., Swin-B: [128, 256, 512, 1024])
        self._chs: List[int] = list(self.body.feature_info.channels())
        assert len(self._chs) == len(out_indices), "out_indices and channels() must match"

        self.fpn = FeaturePyramidNetwork(in_channels_list=self._chs, out_channels=fpn_out_channels)
        self.out_channels = fpn_out_channels
        self._stage_channels = set(self._chs)

    @staticmethod
    def _maybe_to_nchw(t: torch.Tensor, stage_channels: set[int]) -> torch.Tensor:
        """
        If feature is NHWC (B,H,W,C), permute to NCHW (heuristic based on channel size).
        timm usually provides NCHW - the heuristic is just a safety net.
        """
        if t.ndim == 4:
            b, d1, d2, d3 = t.shape
            if (d1 not in stage_channels) and (d3 in stage_channels):
                return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # NOTE: x MUST be (N,3,640,640) - otherwise AssertionError in PatchEmbed (timm).
        feats_list = self.body(x)  # list[Tensor], stages 0..3
        feats_list = [self._maybe_to_nchw(t, self._stage_channels) for t in feats_list]
        feats_in = OrderedDict((str(i), t) for i, t in enumerate(feats_list))
        return self.fpn(feats_in)


def get_model(
    num_classes: int,
    pretrained: bool = True,
) -> FasterRCNN:
    # Swin-Backbone (fixed 640x640)
    backbone = TimmSwinBackbone(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        out_indices=(0, 1, 2, 3),
        fpn_out_channels=256,
        img_size=(640, 640),
    )

    # Anchors: stable starting point for 640x640, mixed object sizes
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0, 3.0),) * 4,
    )

    # ROI Align
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        # IMPORTANT: We want 640x640 - Transform would otherwise try to resize.
        # Set min_size=max_size=640. (Note: For NON-square inputs
        # torchvision would otherwise only set the short side to 640 -> Therefore
        # your preprocessing MUST deliver 640x640.)
        min_size=640,
        max_size=640,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        # more conservative, stable defaults
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_fg_iou_thresh=0.4,
        rpn_bg_iou_thresh=0.1,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_batch_size_per_image=256,
    )

    # Set predictor properly (num_classes including Background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_model(num_classes: int, pretrained: bool = True):
    """
    Factory for training scripts.
    Expects 640x640 images at input (see get_input_size()).
    """
    return get_model(num_classes=num_classes, pretrained=pretrained)


def get_input_size() -> Tuple[int, int]:
    """For your dataset preprocessing: Target size (H, W) that this model expects."""
    return 640, 640


def get_model_need():
    return "Tensor"


if __name__ == "__main__":
    # Quick test with dummy input (exactly 640x640)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(num_classes=4, pretrained=True).to(device)
    m.eval()
    with torch.no_grad():
        x = torch.rand(3, 640, 640, device=device)  # [0,1], correct size
        preds = m([x])
    summary = {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in preds[0].items()}
    print("Swin Faster R-CNN ready:", summary)
