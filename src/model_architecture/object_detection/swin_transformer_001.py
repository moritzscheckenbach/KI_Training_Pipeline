import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, H, W):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Window partition
        x_windows = self.window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class SwinBackbone(nn.Module):
    """Swin Transformer Backbone for Object Detection"""

    def __init__(self, img_size=224, patch_size=4, embed_dim=128, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=7):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(0.1)

        # Build layers
        self.layers = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            layer = nn.ModuleList([SwinTransformerBlock(dim=embed_dim * (2**i), num_heads=num_head, window_size=window_size) for _ in range(depth)])
            self.layers.append(layer)

            # Patch merging (except for the last layer)
            if i < len(depths) - 1:
                self.layers.append(nn.Conv2d(embed_dim * (2**i), embed_dim * (2 ** (i + 1)), kernel_size=2, stride=2))

        # Feature dimensions after each stage
        self.feature_dims = [embed_dim * (2**i) for i in range(len(depths))]

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.pos_drop(x)

        features = []

        # Forward through Swin blocks
        current_dim = self.embed_dim
        current_H, current_W = H, W

        for i, layer_group in enumerate(self.layers):
            if isinstance(layer_group, nn.ModuleList):
                # Swin Transformer blocks
                for block in layer_group:
                    x = block(x, current_H, current_W)
                features.append(x.transpose(1, 2).view(B, current_dim, current_H, current_W))
            else:
                # Patch merging (downsampling)
                x = x.transpose(1, 2).view(B, current_dim, current_H, current_W)
                x = layer_group(x)  # Conv2d downsampling
                B, current_dim, current_H, current_W = x.shape
                x = x.flatten(2).transpose(1, 2)

        return features[-1]  # Return last feature map


class ObjectDetectionHead(nn.Module):
    """Object Detection Head für Swin Transformer"""

    def __init__(self, in_channels, num_classes=20, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.boxes_per_cell = 2

        # Adaptive pooling to fixed grid size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Detection head
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Output: [batch, grid, grid, (boxes_per_cell * 5 + num_classes)]
        output_size = self.boxes_per_cell * 5 + num_classes  # 5 = (x, y, w, h, confidence)
        self.final_conv = nn.Conv2d(256, output_size, 1)

    def forward(self, x):
        # x shape: [batch, channels, H, W]
        x = self.adaptive_pool(x)  # [batch, channels, grid_size, grid_size]
        x = self.conv_layers(x)
        x = self.final_conv(x)  # [batch, output_size, grid_size, grid_size]

        return x.permute(0, 2, 3, 1)  # [batch, grid_size, grid_size, output_size]


class SwinTransformerObjectDetection(nn.Module):
    """Swin Transformer für Object Detection - Pipeline-kompatibel"""

    def __init__(self, img_size=224, num_classes=20, patch_size=4, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.grid_size = 7

        # Swin Transformer Backbone
        self.backbone = SwinBackbone(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32])

        # Object Detection Head
        final_dim = embed_dim * 8  # Nach 3 Downsamplings: 128 * 2^3
        self.detection_head = ObjectDetectionHead(in_channels=final_dim, num_classes=num_classes, grid_size=self.grid_size)

    def forward(self, x, targets=None):
        # Backbone forward
        features = self.backbone(x)  # [batch, channels, H, W]

        # Detection head forward
        detection_output = self.detection_head(features)  # [batch, grid, grid, output_size]

        if targets is not None:
            # Training mode - compute losses
            return self._compute_losses(detection_output, targets)
        else:
            # Inference mode
            return self._parse_detections(detection_output)

    def _compute_losses(self, predictions, targets):
        """Compute YOLO-style losses"""
        batch_size = predictions.size(0)
        grid_size = self.grid_size

        # Initialize losses
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0

        lambda_coord = 5.0
        lambda_noobj = 0.5

        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]  # [grid, grid, output_size]
            target_dict = targets[batch_idx]

            if len(target_dict.get("boxes", [])) == 0:
                # No objects - only no-object confidence loss
                pred_conf = pred[:, :, 4::5]  # Confidence predictions
                no_obj_loss = F.mse_loss(pred_conf, torch.zeros_like(pred_conf))
                conf_loss += lambda_noobj * no_obj_loss
                continue

            # Process targets
            boxes = target_dict["boxes"]  # [num_objects, 4]
            labels = target_dict["labels"]  # [num_objects]

            # Create target tensor
            target_tensor = torch.zeros_like(pred)

            for obj_idx, (box, label) in enumerate(zip(boxes, labels)):
                # Convert box to grid coordinates
                x_center = box[0] + box[2] / 2
                y_center = box[1] + box[3] / 2
                width = box[2]
                height = box[3]

                # Normalize to [0, 1]
                x_center /= self.img_size
                y_center /= self.img_size
                width /= self.img_size
                height /= self.img_size

                # Find grid cell
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)

                # Relative position within grid cell
                x_offset = (x_center * grid_size) - grid_x
                y_offset = (y_center * grid_size) - grid_y

                # Set target values
                target_tensor[grid_y, grid_x, 0] = x_offset  # x
                target_tensor[grid_y, grid_x, 1] = y_offset  # y
                target_tensor[grid_y, grid_x, 2] = width  # w
                target_tensor[grid_y, grid_x, 3] = height  # h
                target_tensor[grid_y, grid_x, 4] = 1.0  # confidence

                # Class probabilities
                class_start_idx = 10  # After 2 boxes * 5 values each
                target_tensor[grid_y, grid_x, class_start_idx + label] = 1.0

            # Compute losses
            # Coordinate loss
            obj_mask = target_tensor[:, :, 4] > 0  # Cells with objects
            if obj_mask.sum() > 0:
                pred_coords = pred[obj_mask][:, :4]
                target_coords = target_tensor[obj_mask][:, :4]
                coord_loss += lambda_coord * F.mse_loss(pred_coords, target_coords)

                # Object confidence loss
                pred_conf_obj = pred[obj_mask][:, 4]
                target_conf_obj = target_tensor[obj_mask][:, 4]
                conf_loss += F.mse_loss(pred_conf_obj, target_conf_obj)

                # Classification loss
                class_start_idx = 10
                pred_class = pred[obj_mask][:, class_start_idx:]
                target_class = target_tensor[obj_mask][:, class_start_idx:]
                class_loss += F.mse_loss(pred_class, target_class)

            # No-object confidence loss
            noobj_mask = target_tensor[:, :, 4] == 0
            if noobj_mask.sum() > 0:
                pred_conf_noobj = pred[noobj_mask][:, 4]
                target_conf_noobj = target_tensor[noobj_mask][:, 4]
                conf_loss += lambda_noobj * F.mse_loss(pred_conf_noobj, target_conf_noobj)

        # Average over batch
        coord_loss /= batch_size
        conf_loss /= batch_size
        class_loss /= batch_size

        return {"bbox_regression_loss": coord_loss, "confidence_loss": conf_loss, "classification_loss": class_loss}

    def _parse_detections(self, predictions):
        """Parse grid predictions to detections"""
        batch_size = predictions.size(0)
        grid_size = self.grid_size

        all_detections = []

        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]  # [grid, grid, output_size]
            detections = []

            for i in range(grid_size):
                for j in range(grid_size):
                    cell_pred = pred[i, j]

                    # Extract box predictions (using first box for simplicity)
                    x_offset = torch.sigmoid(cell_pred[0])
                    y_offset = torch.sigmoid(cell_pred[1])
                    width = cell_pred[2]
                    height = cell_pred[3]
                    confidence = torch.sigmoid(cell_pred[4])

                    # Convert to absolute coordinates
                    x_center = (j + x_offset) / grid_size
                    y_center = (i + y_offset) / grid_size

                    # Convert to corner coordinates
                    x1 = (x_center - width / 2) * self.img_size
                    y1 = (y_center - height / 2) * self.img_size
                    x2 = (x_center + width / 2) * self.img_size
                    y2 = (y_center + height / 2) * self.img_size

                    # Class predictions
                    class_start_idx = 10
                    class_probs = torch.softmax(cell_pred[class_start_idx:], dim=0)
                    class_conf, class_pred = torch.max(class_probs, dim=0)

                    final_conf = confidence * class_conf

                    if final_conf > 0.5:  # Confidence threshold
                        detections.append({"bbox": [x1.item(), y1.item(), x2.item(), y2.item()], "score": final_conf.item(), "class": class_pred.item()})

            all_detections.append(detections)

        return {"predictions": all_detections}

    def get_input_size(self):
        """Return expected input size as (height, width)"""
        return (self.img_size, self.img_size)


def build_model(num_classes=20, img_size=224):
    """Factory function für Swin Transformer Object Detection"""
    model = SwinTransformerObjectDetection(img_size=img_size, num_classes=num_classes, patch_size=4, embed_dim=128)
    return model
