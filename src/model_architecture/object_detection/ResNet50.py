import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50ObjectDetection(nn.Module):
    """
    ResNet50-based Object Detection with Transfer Learning
    """

    def __init__(self, num_classes=20, input_size=416, grid_size=13, pretrained=True):
        super(ResNet50ObjectDetection, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.grid_size = grid_size

        # Load pretrained ResNet50 (ohne final layers)
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)

        # Remove final layers (avgpool + fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Freeze early layers für Transfer Learning (optional)
        self._freeze_early_layers()

        # Adaptive pooling um auf grid_size zu kommen
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Detection head
        # ResNet50 hat 2048 features nach dem letzten conv block
        self.output_channels = num_classes + 5  # classes + x,y,w,h + confidence
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.output_channels, kernel_size=1),
        )

        # Initialize nur die neuen layers
        self._initialize_new_layers()

    def _freeze_early_layers(self, freeze_until_layer=6):
        """Freeze early layers für Transfer Learning"""
        children = list(self.backbone.children())
        for i, child in enumerate(children):
            if i < freeze_until_layer:
                for param in child.parameters():
                    param.requires_grad = False

    def _initialize_new_layers(self):
        """Initialize nur die detection head weights"""
        for module in self.detection_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, targets=None):
        # Backbone
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        # Head
        detection_output = self.detection_head(features)

        if self.training:
            # Erwartet: targets ist gesetzt
            if targets is None:
                raise ValueError("targets must be provided during training")
            return self._compute_losses(detection_output, targets)
        else:
            # Inference
            predictions = self._parse_detections(detection_output)  # <- gibt bei dir offenbar None zurück

            B = x.shape[0]
            device = x.device
            # Robustheit: leere Detections je Bild, falls None oder falsches Format
            if predictions is None or not isinstance(predictions, (list, tuple)) or len(predictions) != B:
                empty = {
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                }
                return [empty.copy() for _ in range(B)]

            # Optional: sicherstellen, dass keys existieren
            norm = []
            for p in predictions:
                norm.append(
                    {
                        "boxes": p.get("boxes", torch.empty((0, 4), device=device)),
                        "scores": p.get("scores", torch.empty((0,), device=device)),
                        "labels": p.get("labels", torch.empty((0,), dtype=torch.long, device=device)),
                    }
                )
            return norm

    def _compute_losses(self, detection_output, targets):
        """Compute training losses (kopiert von cnn_001)"""
        device = detection_output.device
        batch_size = detection_output.size(0)

        total_conf_loss = 0.0
        total_class_loss = 0.0
        total_bbox_loss = 0.0

        for i in range(batch_size):
            target = targets[i]
            pred = detection_output[i]

            if len(target["labels"]) == 0:
                conf_pred = torch.sigmoid(pred[-1])
                conf_target = torch.zeros_like(conf_pred)
                total_conf_loss += F.mse_loss(conf_pred, conf_target)
                continue

            # Erstelle Grid-Targets
            class_target = torch.zeros((self.num_classes, 13, 13), device=device)
            bbox_target = torch.zeros((4, 13, 13), device=device)
            conf_target = torch.zeros((13, 13), device=device)

            # Für jedes Ground Truth Objekt
            for j in range(len(target["labels"])):
                label = target["labels"][j]
                bbox = target["boxes"][j]  # [x, y, w, h] normalisiert

                # Finde Grid-Zelle
                grid_x = int(bbox[0] / self.input_size * 13)
                grid_y = int(bbox[1] / self.input_size * 13)
                grid_x = min(grid_x, 12)
                grid_y = min(grid_y, 12)

                # Setze Targets
                class_target[label, grid_y, grid_x] = 1.0
                bbox_target[:, grid_y, grid_x] = bbox
                conf_target[grid_y, grid_x] = 1.0

            # Berechne Losses
            # Class Loss
            class_pred = torch.softmax(pred[: self.num_classes], dim=0)
            class_loss = F.mse_loss(class_pred, class_target)

            # Bbox Loss
            bbox_pred = pred[self.num_classes : self.num_classes + 4]
            bbox_loss = F.mse_loss(bbox_pred * conf_target.unsqueeze(0), bbox_target * conf_target.unsqueeze(0))

            # Confidence Loss
            conf_pred = torch.sigmoid(pred[-1])  # ✅ Letzter Kanal ist confidence
            conf_loss = F.mse_loss(conf_pred, conf_target)

            total_class_loss += class_loss
            total_bbox_loss += bbox_loss
            total_conf_loss += conf_loss

        return {"classification_loss": total_class_loss / batch_size, "bbox_regression_loss": total_bbox_loss / batch_size, "confidence_loss": total_conf_loss / batch_size}

    def _parse_detections_old(self, detection_output):
        """Parse grid output to detections"""
        batch_size = detection_output.size(0)
        detections = []

        for i in range(batch_size):
            pred = detection_output[i]  # [output_channels, 13, 13]

            # Extract predictions
            class_pred = torch.softmax(pred[: self.num_classes], dim=0)
            bbox_pred = pred[self.num_classes : self.num_classes + 4]
            conf_pred = torch.sigmoid(pred[-1])

            # Find detections above threshold
            conf_mask = conf_pred > 0.5

            if conf_mask.sum() == 0:
                detections.append({"class_scores": torch.empty((0, self.num_classes)), "bbox_coords": torch.empty((0, 4)), "confidence": torch.empty((0,))})
                continue

            # Get detections
            y_coords, x_coords = torch.where(conf_mask)

            detected_classes = class_pred[:, y_coords, x_coords].t()  # [num_detections, num_classes]
            detected_boxes = bbox_pred[:, y_coords, x_coords].t()  # [num_detections, 4]
            detected_conf = conf_pred[y_coords, x_coords]  # [num_detections]

            detections.append({"class_scores": detected_classes, "bbox_coords": detected_boxes, "confidence": detected_conf})

        return detections

    def _parse_detections(self, detection_output):
        """Parse grid output to COCO-format detections"""
        batch_size = detection_output.size(0)
        device = detection_output.device
        detections = []

        for i in range(batch_size):
            pred = detection_output[i]  # [output_channels, grid_size, grid_size]

            # Extract predictions
            class_pred = pred[: self.num_classes]  # [num_classes, grid, grid]
            bbox_pred = pred[self.num_classes : self.num_classes + 4]  # [4, grid, grid]
            conf_pred = torch.sigmoid(pred[-1])  # [grid, grid] - FIXED: using last channel

            # Find detections above threshold
            conf_mask = conf_pred > 0.1  # Lower threshold during early training

            if not conf_mask.any():
                # No detections found, return empty detection
                empty = {
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                }
                detections.append(empty)
                continue

            # Get detection coordinates
            y_coords, x_coords = torch.where(conf_mask)

            # Get class predictions at these coordinates
            class_scores = class_pred[:, y_coords, x_coords]  # [num_classes, num_detections]
            pred_classes = torch.argmax(class_scores, dim=0)  # [num_detections]
            max_scores = torch.max(class_scores, dim=0)[0]  # [num_detections]

            # Get confidence scores
            confidence = conf_pred[y_coords, x_coords]  # [num_detections]

            # Combine confidence and class score for final score
            scores = confidence * max_scores

            # Get bounding box coordinates
            boxes = bbox_pred[:, y_coords, x_coords].t()  # [num_detections, 4]

            # Convert grid-relative coordinates to image space
            # Add grid offset and scale to image size
            grid_size = self.grid_size
            cell_size = 1.0 / grid_size

            # Convert [x, y, w, h] relative format to [x1, y1, x2, y2] absolute format
            x_offset = x_coords.float() * cell_size
            y_offset = y_coords.float() * cell_size

            # Add grid cell offset to x,y and scale to input size
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (x_offset + boxes[:, 0] * cell_size) * self.input_size  # x1
            boxes_xyxy[:, 1] = (y_offset + boxes[:, 1] * cell_size) * self.input_size  # y1
            boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes[:, 2] * self.input_size  # x2 = x1 + w
            boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes[:, 3] * self.input_size  # y2 = y1 + h

            # Return in COCO format
            detections.append(
                {
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": pred_classes,
                }
            )

        return detections


def get_input_size():
    """Return expected input size as (height, width)"""
    return (416, 416)


def build_model(num_classes=20, model_type="full"):
    """
    Factory function to build the ResNet50 model
    """
    return ResNet50ObjectDetection(num_classes=num_classes, pretrained=True)


def get_model_info():
    """
    Returns model information
    """
    return {
        "name": "ResNet50ObjectDetection",
        "type": "Object Detection with Transfer Learning",
        "backbone": "ResNet50 (ImageNet pretrained)",
        "input_size": (3, 416, 416),
        "outputs": {"class_scores": "Classification logits", "bbox_coords": "Bounding box coordinates [x, y, w, h]"},
    }


# Test function
if __name__ == "__main__":
    model = build_model(num_classes=20)
    model.eval()
    dummy_input = torch.randn(1, 3, 416, 416)
    output = model(dummy_input)
    print("ResNet50 Model loaded successfully!")
    print(f"Input size: {get_input_size()}")
    print(f"Dummy output: {output}")
