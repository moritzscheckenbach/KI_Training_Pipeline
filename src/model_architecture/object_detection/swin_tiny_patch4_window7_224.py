import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinTinyObjectDetection(nn.Module):
    """Swin Tiny mit vortrainiertem Backbone für Object Detection"""
    
    def __init__(self, num_classes=20, pretrained=True, grid_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.boxes_per_cell = 2
        
        # Load pretrained Swin Tiny from timm
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=[3]      # Use last stage features (highest level)
        )
        
        # Get feature dimensions from backbone - check the actual output
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            backbone_channels = dummy_features[0].shape[1]
        
        # Object Detection Head (YOLO-style)
        self.detection_head = ObjectDetectionHead(
            in_channels=backbone_channels,
            num_classes=num_classes,
            grid_size=grid_size
        )
        
        # Freeze backbone initially (optional for transfer learning)
        self.freeze_backbone = False
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x, targets=None):
        # Extract features using Swin Tiny backbone
        features = self.backbone(x)  # Returns list of feature maps
        feature_map = features[0]    # [batch, channels, H, W]
        
        # Object detection head
        detection_output = self.detection_head(feature_map)  # [batch, grid, grid, output_size]
        
        if targets is not None:
            # Training mode - compute losses
            return self._compute_losses(detection_output, targets)
        else:
            # Inference mode
            return self._parse_detections(detection_output)
    
    def _compute_losses(self, predictions, targets):
        """YOLO-style loss computation"""
        batch_size = predictions.size(0)
        grid_size = self.grid_size
        device = predictions.device
        
        # Loss coefficients
        lambda_coord = 5.0
        lambda_noobj = 0.5
        
        # Initialize losses
        coord_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)
        
        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]  # [grid, grid, output_size]
            target_dict = targets[batch_idx]
            
            # Handle empty targets
            if len(target_dict.get('boxes', [])) == 0:
                # Only no-object confidence loss
                pred_conf = pred[:, :, 4::5]  # Extract confidence predictions
                no_obj_target = torch.zeros_like(pred_conf)
                conf_loss += lambda_noobj * F.mse_loss(pred_conf, no_obj_target)
                continue
            
            # Process targets
            boxes = target_dict['boxes'].to(device)    # [num_objects, 4]
            labels = target_dict['labels'].to(device)  # [num_objects]
            
            # Create target tensor
            target_tensor = torch.zeros_like(pred)
            
            # Convert COCO format [x, y, width, height] to grid targets
            img_size = 224  # Swin Tiny input size
            
            for box, label in zip(boxes, labels):
                # COCO format: [x_min, y_min, width, height]
                x_center = box[0] + box[2] / 2
                y_center = box[1] + box[3] / 2
                width = box[2]
                height = box[3]
                
                # Normalize to [0, 1]
                x_center /= img_size
                y_center /= img_size
                width /= img_size
                height /= img_size
                
                # Find responsible grid cell
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)
                
                # Relative position within cell
                x_offset = (x_center * grid_size) - grid_x
                y_offset = (y_center * grid_size) - grid_y
                
                # Set target for first bounding box predictor
                target_tensor[grid_y, grid_x, 0] = x_offset
                target_tensor[grid_y, grid_x, 1] = y_offset
                target_tensor[grid_y, grid_x, 2] = torch.sqrt(width) if width > 0 else 0
                target_tensor[grid_y, grid_x, 3] = torch.sqrt(height) if height > 0 else 0
                target_tensor[grid_y, grid_x, 4] = 1.0  # Confidence
                
                # Class probabilities (one-hot)
                class_start_idx = self.boxes_per_cell * 5
                if label < self.num_classes:
                    target_tensor[grid_y, grid_x, class_start_idx + label] = 1.0
            
            # Compute losses
            obj_mask = target_tensor[:, :, 4] > 0  # Cells with objects
            
            if obj_mask.sum() > 0:
                # Coordinate loss (only for cells with objects)
                pred_coords = pred[obj_mask][:, :4]
                target_coords = target_tensor[obj_mask][:, :4]
                
                # Apply square root to width/height predictions
                pred_coords_adj = pred_coords.clone()
                pred_coords_adj[:, 2] = torch.sqrt(torch.abs(pred_coords[:, 2]) + 1e-8)
                pred_coords_adj[:, 3] = torch.sqrt(torch.abs(pred_coords[:, 3]) + 1e-8)
                
                coord_loss += lambda_coord * F.mse_loss(pred_coords_adj, target_coords)
                
                # Object confidence loss
                pred_conf_obj = torch.sigmoid(pred[obj_mask][:, 4])
                target_conf_obj = target_tensor[obj_mask][:, 4]
                conf_loss += F.binary_cross_entropy(pred_conf_obj, target_conf_obj, reduction='mean')
                
                # Classification loss
                class_start_idx = self.boxes_per_cell * 5
                pred_class = pred[obj_mask][:, class_start_idx:class_start_idx + self.num_classes]
                target_class = target_tensor[obj_mask][:, class_start_idx:class_start_idx + self.num_classes]
                class_loss += F.binary_cross_entropy_with_logits(pred_class, target_class, reduction='mean')
            
            # No-object confidence loss
            noobj_mask = target_tensor[:, :, 4] == 0
            if noobj_mask.sum() > 0:
                pred_conf_noobj = torch.sigmoid(pred[noobj_mask][:, 4])
                target_conf_noobj = target_tensor[noobj_mask][:, 4]
                conf_loss += lambda_noobj * F.binary_cross_entropy(pred_conf_noobj, target_conf_noobj, reduction='mean')
        
        # Average over batch
        coord_loss = coord_loss / batch_size if batch_size > 0 else coord_loss
        conf_loss = conf_loss / batch_size if batch_size > 0 else conf_loss
        class_loss = class_loss / batch_size if batch_size > 0 else class_loss
        
        return {
            "bbox_regression_loss": coord_loss,
            "confidence_loss": conf_loss,
            "classification_loss": class_loss
        }
    
    def _parse_detections(self, predictions):
        """Convert grid predictions to bounding box detections"""
        batch_size = predictions.size(0)
        grid_size = self.grid_size
        img_size = 224
        confidence_threshold = 0.3
        
        all_detections = []
        
        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]  # [grid, grid, output_size]
            batch_detections = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_pred = pred[i, j]
                    
                    # Extract first bounding box prediction
                    x_offset = torch.sigmoid(cell_pred[0])
                    y_offset = torch.sigmoid(cell_pred[1])
                    width = cell_pred[2] ** 2  # Reverse square root
                    height = cell_pred[3] ** 2
                    confidence = torch.sigmoid(cell_pred[4])
                    
                    # Convert to absolute coordinates
                    x_center = (j + x_offset) / grid_size
                    y_center = (i + y_offset) / grid_size
                    
                    # Convert to COCO format [x_min, y_min, width, height]
                    x_min = (x_center - width/2) * img_size
                    y_min = (y_center - height/2) * img_size
                    box_width = width * img_size
                    box_height = height * img_size
                    
                    # Class predictions
                    class_start_idx = self.boxes_per_cell * 5
                    class_logits = cell_pred[class_start_idx:class_start_idx + self.num_classes]
                    class_probs = torch.softmax(class_logits, dim=0)
                    class_conf, class_pred = torch.max(class_probs, dim=0)
                    
                    final_conf = confidence * class_conf
                    
                    if final_conf > confidence_threshold:
                        batch_detections.append({
                            'bbox': [x_min.item(), y_min.item(), box_width.item(), box_height.item()],
                            'score': final_conf.item(),
                            'category_id': class_pred.item()
                        })
            
            all_detections.append(batch_detections)
        
        return {"predictions": all_detections}
    
    def get_input_size(self):
        """Return expected input size (height, width)"""
        return (224, 224)  # Swin Tiny standard input size
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

class ObjectDetectionHead(nn.Module):
    """Detection head for converting Swin features to YOLO-style predictions"""
    
    def __init__(self, in_channels, num_classes=20, grid_size=7, boxes_per_cell=2):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell
        
        # First, adapt the channels to a standard size
        self.channel_adapter = nn.Conv2d(in_channels, 512, 1)
        
        # Upsample features to grid size if needed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # Detection layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Output layer: boxes_per_cell * (4 coords + 1 confidence) + num_classes
        output_size = boxes_per_cell * 5 + num_classes
        self.final_conv = nn.Conv2d(256, output_size, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch, channels, H, W] from Swin backbone
        x = self.channel_adapter(x)    # [batch, 512, H, W]
        x = self.adaptive_pool(x)      # [batch, 512, grid_size, grid_size]
        x = self.conv_layers(x)        # [batch, 256, grid_size, grid_size]
        x = self.final_conv(x)         # [batch, output_size, grid_size, grid_size]
        
        return x.permute(0, 2, 3, 1)  # [batch, grid_size, grid_size, output_size]

def build_model(config=None):
    """Factory function for Swin Tiny Object Detection"""
    num_classes = 20
    pretrained = True
    
    if config:
        num_classes = config.get("num_classes", 20)
        pretrained = config.get("pretrained", True)
    
    model = SwinTinyObjectDetection(
        num_classes=num_classes,
        pretrained=pretrained,
        grid_size=7
    )
    
    return model