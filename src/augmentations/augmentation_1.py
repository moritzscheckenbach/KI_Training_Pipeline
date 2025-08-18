# augment_01.py
import torch
from torchvision.transforms import v2


def augment():
    """
    Minimal v2-Pipeline, bbox-aware.
    - ToImage: PIL -> tv_tensors.Image (CHW)
    - ToDtype: float32 [0,1]
    - Leichte Augmentierungen (kompatibel zu BBox-Tracking)
    - SanitizeBoundingBoxes: entfernt/clippt invalide Boxen

    Resize wird absichtlich NICHT hier gesetzt, damit die Modell-Inputgröße
    weiterhin im Dataset/Training bestimmt werden kann.
    """

    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
            ),
            v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            v2.SanitizeBoundingBoxes(),
        ]
    )
