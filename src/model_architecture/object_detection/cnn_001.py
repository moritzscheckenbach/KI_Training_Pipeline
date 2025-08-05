import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for Object Detection
    Outputs class predictions and bounding box coordinates
    """

    def __init__(self, num_classes=20, input_size=416):
        super(SimpleCNN, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 416 -> 208
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 208 -> 104
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 104 -> 52
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52 -> 26
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 26 -> 13
        )

        # Calculate the flattened feature size
        # After all pooling: 416 -> 208 -> 104 -> 52 -> 26 -> 13
        self.feature_size = 512 * 13 * 13

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.feature_size, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),  # 4 coordinates: x, y, width, height
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Feature extraction
        features = self.features(x)

        # Flatten features
        features = features.view(features.size(0), -1)

        # Classification output
        class_scores = self.classifier(features)

        # Bounding box output
        bbox_coords = self.bbox_regressor(features)

        return {"class_scores": class_scores, "bbox_coords": bbox_coords}

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


class SimpleCNNClassificationOnly(nn.Module):
    """
    Simplified version for classification only (if you only need classes)
    """

    def __init__(self, num_classes=20):
        super(SimpleCNNClassificationOnly, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_model(num_classes=20, model_type="full"):
    """
    Factory function to build the model

    Args:
        num_classes: Number of object classes
        model_type: "full" for detection, "classification" for classification only

    Returns:
        CNN model
    """
    if model_type == "classification":
        return SimpleCNNClassificationOnly(num_classes=num_classes)
    else:
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
