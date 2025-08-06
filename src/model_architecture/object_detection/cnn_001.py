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
        # Feature extraction
        features = self.features(x)
        detection_output = self.detection_head(features)
        
        if targets is not None:
            # Training/Validation: Return losses
            return self._compute_losses(detection_output, targets)
        else:
            # Inference: Return predictions (but still as dict!)
            predictions = self._parse_detections(detection_output)
            return {
                "predictions": predictions,
                "classification_loss": torch.tensor(0.0, device=x.device),
                "bbox_regression_loss": torch.tensor(0.0, device=x.device),
                "confidence_loss": torch.tensor(0.0, device=x.device)
            }

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
                conf_pred = torch.sigmoid(pred[4])  # Confidence channel
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
            class_pred = torch.softmax(pred[:self.num_classes], dim=0)
            class_loss = F.mse_loss(class_pred, class_target)
            
            # Bbox Loss
            bbox_pred = pred[self.num_classes:self.num_classes+4]
            bbox_loss = F.mse_loss(bbox_pred * conf_target.unsqueeze(0), 
                                 bbox_target * conf_target.unsqueeze(0))
            
            # Confidence Loss
            conf_pred = torch.sigmoid(pred[4])
            conf_loss = F.mse_loss(conf_pred, conf_target)
            
            total_class_loss += class_loss
            total_bbox_loss += bbox_loss
            total_conf_loss += conf_loss
        
        return {
            "classification_loss": total_class_loss / batch_size,
            "bbox_regression_loss": total_bbox_loss / batch_size,
            "confidence_loss": total_conf_loss / batch_size
        }

    def _parse_detections(self, detection_output):
        """Parse grid output to detections"""
        batch_size = detection_output.size(0)
        detections = []
        
        for i in range(batch_size):
            pred = detection_output[i]  # [output_channels, 13, 13]
            
            # Extract predictions
            class_pred = torch.softmax(pred[:self.num_classes], dim=0)
            bbox_pred = pred[self.num_classes:self.num_classes+4]
            conf_pred = torch.sigmoid(pred[4])
            
            # Find detections above threshold
            conf_mask = conf_pred > 0.5
            
            if conf_mask.sum() == 0:
                detections.append({
                    "class_scores": torch.empty((0, self.num_classes)),
                    "bbox_coords": torch.empty((0, 4)),
                    "confidence": torch.empty((0,))
                })
                continue
            
            # Get detections
            y_coords, x_coords = torch.where(conf_mask)
            
            detected_classes = class_pred[:, y_coords, x_coords].t()  # [num_detections, num_classes]
            detected_boxes = bbox_pred[:, y_coords, x_coords].t()     # [num_detections, 4]
            detected_conf = conf_pred[y_coords, x_coords]            # [num_detections]
            
            detections.append({
                "class_scores": detected_classes,
                "bbox_coords": detected_boxes,
                "confidence": detected_conf
            })
        
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



    def get_input_size(self):

        return (self.input_size, self.input_size)



def build_model(num_classes=20, model_type="full"):
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
    return {"name": "SimpleCNN", "type": "Object Detection", "input_size": (3, 416, 416), "outputs": {"class_scores": "Classification logits", "bbox_coords": "Bounding box coordinates [x, y, w, h]"}}


# Test function
if __name__ == "__main__":
    # Test the model
    model = build_model(num_classes=20)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 416, 416)

    # Forward pass
    output = model(dummy_input)

    print("Model Output Shapes:")
    print(f"Class scores: {output['class_scores'].shape}")
    print(f"Bbox coords: {output['bbox_coords'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
