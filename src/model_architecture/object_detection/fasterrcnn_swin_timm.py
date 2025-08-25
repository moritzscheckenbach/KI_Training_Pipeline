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
    Swin-Backbone aus timm + FPN für Faster R-CNN.
    Variante B: img_size wird beim Erstellen fest auf (640, 640) gesetzt.
    => Eingaben an das Modell MÜSSEN 640x640 sein (sonst AssertionError in PatchEmbed).

    Rückgabe: Dict[str, Tensor] mit Keys "0","1","2","3" (kompatibel zu ROIAlign).
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",  # timm-ID, die existiert
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        fpn_out_channels: int = 256,
        img_size: Tuple[int, int] = (640, 640),  # <- feste Zielgröße
    ):
        super().__init__()
        # timm Feature-Extraktor mit fester img_size
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,  # WICHTIG: erzwingt PatchEmbed-Assertion auf 640x640
        )

        # Kanalzahlen je Stufe (z.B. Swin-B: [128, 256, 512, 1024])
        self._chs: List[int] = list(self.body.feature_info.channels())
        assert len(self._chs) == len(out_indices), "out_indices und channels() müssen matchen"

        self.fpn = FeaturePyramidNetwork(in_channels_list=self._chs, out_channels=fpn_out_channels)
        self.out_channels = fpn_out_channels
        self._stage_channels = set(self._chs)

    @staticmethod
    def _maybe_to_nchw(t: torch.Tensor, stage_channels: set[int]) -> torch.Tensor:
        """
        Falls Feature NHWC ist (B,H,W,C), nach NCHW permutieren (heuristisch anhand Kanalgröße).
        timm liefert i.d.R. NCHW – die Heuristik ist nur ein Sicherheitsnetz.
        """
        if t.ndim == 4:
            b, d1, d2, d3 = t.shape
            if (d1 not in stage_channels) and (d3 in stage_channels):
                return t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # HINWEIS: x MUSS (N,3,640,640) sein – ansonsten AssertionError in PatchEmbed (timm).
        feats_list = self.body(x)  # list[Tensor], Stufen 0..3
        feats_list = [self._maybe_to_nchw(t, self._stage_channels) for t in feats_list]
        feats_in = OrderedDict((str(i), t) for i, t in enumerate(feats_list))
        return self.fpn(feats_in)


def get_model(
    num_classes: int,
    pretrained: bool = True,
) -> FasterRCNN:
    # Swin-Backbone (fest 640x640)
    backbone = TimmSwinBackbone(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        out_indices=(0, 1, 2, 3),
        fpn_out_channels=256,
        img_size=(640, 640),
    )

    # Anchors: stabiler Start für 640x640, gemischte Objektgrößen
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
        # WICHTIG: Wir wollen 640x640 – Transform versucht sonst zu resizen.
        # Setze min_size=max_size=640. (Achtung: Bei NICHT quadratischen Eingaben
        # würde torchvision sonst nur die kurze Seite auf 640 setzen -> Daher
        # MUSS dein Preprocessing bereits 640x640 liefern.)
        min_size=640,
        max_size=640,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        # etwas konservativere, stabile Defaults
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

    # Predictor sauber setzen (num_classes inkl. Background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_model(num_classes: int, pretrained: bool = True):
    """
    Factory für Trainings-Skripte.
    Erwartet 640x640-Bilder am Eingang (siehe get_input_size()).
    """
    return get_model(num_classes=num_classes, pretrained=pretrained)


def get_input_size() -> Tuple[int, int]:
    """Für dein Dataset-Preprocessing: Zielgröße (H, W), die dieses Modell erwartet."""
    return 640, 640


if __name__ == "__main__":
    # Kurztest mit Dummy-Input (exakt 640x640)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(num_classes=4, pretrained=True).to(device)
    m.eval()
    with torch.no_grad():
        x = torch.rand(3, 640, 640, device=device)  # [0,1], richtige Größe
        preds = m([x])
    summary = {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in preds[0].items()}
    print("Swin Faster R-CNN ready:", summary)
