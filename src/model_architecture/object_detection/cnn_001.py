import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Multi-Object Detection CNN (Grid-based approach)
    """

    def __init__(self, num_classes=20, input_size=416, grid_size=13):
        super(SimpleCNN, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.grid_size = grid_size  # 13x13 grid

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1-5 (wie gehabt)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 416 -> 208
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 208 -> 104
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 104 -> 52
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52 -> 26
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 26 -> 13
        )

        # Output layers: für jede Grid-Zelle (13x13)
        # Pro Zelle: 1 Objekt mit Klasse + Bounding Box + Confidence
        self.output_channels = num_classes + 5  # classes + x,y,w,h + confidence

        self.detection_head = nn.Conv2d(512, self.output_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x, targets=None):
        # Apply normalization for ImageNet pretrained features
        if not self.training or targets is None:
            # Assuming input is in range [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std

        # Feature extraction
        features = self.features(x)
        detection_output = self.detection_head(features)

        if self.training and targets is not None:
            # Training: Return losses
            return self._compute_losses(detection_output, targets)
        else:
            # Inference: Return predictions in COCO format
            return self._parse_detections(detection_output)

    def _compute_losses(self, detection_output, targets):
        """Compute training losses for grid-based detection"""
        device = detection_output.device
        batch_size = detection_output.size(0)

        total_conf_loss = 0.0
        total_class_loss = 0.0
        total_bbox_loss = 0.0

        for i in range(batch_size):
            target = targets[i]
            pred = detection_output[i]  # [output_channels, 13, 13]

            if len(target["labels"]) == 0:
                # Keine Objekte - nur Confidence Loss (alles sollte 0 sein)
                conf_pred = torch.sigmoid(pred[-1])  # Confidence channel
                conf_target = torch.zeros_like(conf_pred)
                total_conf_loss += F.mse_loss(conf_pred, conf_target)
                continue

            # Erstelle Grid-Targets
            class_target = torch.zeros((self.num_classes, self.grid_size, self.grid_size), device=device)
            bbox_target = torch.zeros((4, self.grid_size, self.grid_size), device=device)
            conf_target = torch.zeros((self.grid_size, self.grid_size), device=device)

            # Für jedes Ground Truth Objekt
            for j in range(len(target["labels"])):
                label = target["labels"][j]
                bbox = target["boxes"][j]  # [x, y, w, h] normalisiert

                # Finde Grid-Zelle
                grid_x = int(bbox[0] / self.input_size * self.grid_size)
                grid_y = int(bbox[1] / self.input_size * self.grid_size)
                grid_x = min(grid_x, self.grid_size - 1)
                grid_y = min(grid_y, self.grid_size - 1)

                # Setze Targets
                class_target[label, grid_y, grid_x] = 1.0
                bbox_target[:, grid_y, grid_x] = bbox
                conf_target[grid_y, grid_x] = 1.0

            # Berechne Losses
            # Class Loss
            class_pred = pred[: self.num_classes]  # No activation during loss calculation
            class_loss = F.cross_entropy(class_pred.view(self.num_classes, -1), class_target.view(self.num_classes, -1))

            # Bbox Loss
            bbox_pred = pred[self.num_classes : self.num_classes + 4]
            bbox_loss = F.mse_loss(bbox_pred * conf_target.unsqueeze(0), bbox_target * conf_target.unsqueeze(0))

            # Confidence Loss
            conf_pred = pred[-1]  # Last channel is confidence
            conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_target)

            total_class_loss += class_loss
            total_bbox_loss += bbox_loss
            total_conf_loss += conf_loss

        return {"classification_loss": total_class_loss / batch_size, "bbox_regression_loss": total_bbox_loss / batch_size, "confidence_loss": total_conf_loss / batch_size}

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
            conf_pred = torch.sigmoid(pred[-1])  # [grid, grid] - confidence in 0-1 range

            # Find detections above threshold
            conf_mask = conf_pred > 0.01  # Lower threshold during early training

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
            max_scores = torch.max(F.softmax(class_scores, dim=0), dim=0)[0]  # [num_detections]

            # Get confidence scores
            confidence = conf_pred[y_coords, x_coords]  # [num_detections]

            # Combine confidence and class score for final score
            scores = confidence * max_scores

            # Get bounding box coordinates
            boxes = bbox_pred[:, y_coords, x_coords].t()  # [num_detections, 4]

            # Grid-related constants
            grid_size = self.grid_size
            cell_size = 1.0 / grid_size

            # Get grid cell offsets
            x_offset = x_coords.float() * cell_size
            y_offset = y_coords.float() * cell_size

            # Apply sigmoid to get normalized coordinates (0-1)
            # For box coordinates, apply sigmoid to center x,y and exp to width/height
            box_xy = torch.sigmoid(boxes[:, :2])  # x,y coordinates (0-1 within cell)
            box_wh = torch.exp(boxes[:, 2:]) * cell_size  # width,height (0-1, bounded by cell size)

            # Add grid cell offset to x,y
            box_xy[:, 0] += x_offset  # add cell x offset
            box_xy[:, 1] += y_offset  # add cell y offset

            # Convert to absolute XYXY format
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (box_xy[:, 0] - box_wh[:, 0] / 2) * self.input_size  # x1
            boxes_xyxy[:, 1] = (box_xy[:, 1] - box_wh[:, 1] / 2) * self.input_size  # y1
            boxes_xyxy[:, 2] = (box_xy[:, 0] + box_wh[:, 0] / 2) * self.input_size  # x2
            boxes_xyxy[:, 3] = (box_xy[:, 1] + box_wh[:, 1] / 2) * self.input_size  # y2

            # Clamp to image boundaries
            boxes_xyxy = torch.clamp(boxes_xyxy, min=0, max=self.input_size)

            # Return in COCO format
            detections.append(
                {
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": pred_classes,
                }
            )

        return detections

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_detections(self, x):
        """Explicit method to get detections for evaluation"""
        # Save original training state
        was_training = self.training
        self.eval()

        # Apply normalization consistently
        if x.max() > 1.0:
            x = x / 255.0

        # Get detections
        with torch.no_grad():
            detections = self.forward(x)

        # Restore original training state
        self.train(was_training)
        return detections


def get_input_size():
    return 416, 416


def get_model_need():
    return "Tensor"


def build_model(cfg=None, num_classes=20, model_type="full"):
    """
    Factory function to build the model

    Args:
        num_classes: Number of object classes
        model_type: "full" for detection, "classification" for classification only

    Returns:
        CNN model
    """
    return SimpleCNN(num_classes=num_classes)


def get_model_info():
    """
    Returns model information
    """
    return {
        "name": "SimpleCNN",
        "type": "Object Detection",
        "input_size": (3, 416, 416),
        "outputs": {"boxes": "Bounding boxes in (x1, y1, x2, y2) format", "scores": "Detection confidence scores", "labels": "Class labels"},
    }
