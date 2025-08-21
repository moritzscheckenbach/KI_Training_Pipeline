import torch
from torchvision.transforms import v2


def augment():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.SanitizeBoundingBoxes(),
        ]
    )
