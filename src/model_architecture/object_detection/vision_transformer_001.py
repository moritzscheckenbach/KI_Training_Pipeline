import torch
import torch.nn as nn


class ViT(nn.Module):
    """Standard Vision Transformer"""

    def __init__(self, dim=128, image_size=224, patch_size=16, num_classes=10, depth=6, heads=8, mlp_dim=256):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3  # 3 Kanäle (RGB)

        self.patch_size = patch_size
        self.image_size = image_size
        self.dim = dim
        self.num_classes = num_classes
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

    def get_input_size(self):
        """Return expected input size as (height, width)"""
        return (self.image_size, self.image_size)


def build_model(num_classes=10, config=None):
    """
    Factory function to build the SimpleViT model
    Default values are used if config is None
    """
    if config is not None:
        dim = config.get("dim", 128)
        image_size = config.get("image_size", 224)
        patch_size = config.get("patch_size", 16)
        depth = config.get("depth", 6)
        heads = config.get("heads", 8)
        mlp_dim = config.get("mlp_dim", 256)
        num_classes = config.get("num_classes", num_classes)
    else:
        dim = 128
        image_size = 224
        patch_size = 16
        depth = 6
        heads = 8
        mlp_dim = 256

    return ViT(dim=dim, image_size=image_size, patch_size=patch_size, heads=heads, num_classes=num_classes, depth=depth, mlp_dim=mlp_dim)
