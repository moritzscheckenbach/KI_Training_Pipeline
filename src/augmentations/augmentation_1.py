import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class ObjectDetectionAugmentation:
    """
    Vollständige Augmentierung für Object Detection
    Alle Parameter sind hier fest definiert
    """

    def __init__(self):
        # Augmentierungs-Parameter (hier fest definiert)
        self.horizontal_flip_prob = 0.5
        self.vertical_flip_prob = 0.1
        self.rotation_degrees = 10
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.hue_range = (-0.1, 0.1)
        self.blur_prob = 0.2
        self.noise_prob = 0.1

    def __call__(self, image):
        """
        Führt alle Augmentierungen durch

        Args:
            image: PIL Image

        Returns:
            Augmentiertes PIL Image
        """
        # Horizontal Flip
        if random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)

        # Vertical Flip (seltener für Object Detection)
        if random.random() < self.vertical_flip_prob:
            image = F.vflip(image)

        # Rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = F.rotate(image, angle, expand=False, fill=0)

        # Color Jittering
        # Brightness
        brightness_factor = random.uniform(*self.brightness_range)
        image = F.adjust_brightness(image, brightness_factor)

        # Contrast
        contrast_factor = random.uniform(*self.contrast_range)
        image = F.adjust_contrast(image, contrast_factor)

        # Saturation
        saturation_factor = random.uniform(*self.saturation_range)
        image = F.adjust_saturation(image, saturation_factor)

        # Hue
        hue_factor = random.uniform(*self.hue_range)
        image = F.adjust_hue(image, hue_factor)

        # Gaussian Blur (optional)
        if random.random() < self.blur_prob:
            # Konvertiere zu Tensor für GaussianBlur
            if isinstance(image, Image.Image):
                image = F.to_tensor(image)
                image = F.gaussian_blur(image, kernel_size=3, sigma=(0.1, 2.0))
                image = F.to_pil_image(image)

        # Noise (optional)
        if random.random() < self.noise_prob:
            if isinstance(image, Image.Image):
                image = F.to_tensor(image)
                noise = torch.randn_like(image) * 0.01
                image = torch.clamp(image + noise, 0, 1)
                image = F.to_pil_image(image)

        return image


def get_train_transforms():
    """
    Erstellt die komplette Transform-Pipeline für Training

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose(
        [
            ObjectDetectionAugmentation(),  # Unsere Custom Augmentierung
            transforms.Resize((416, 416)),  # Standardgröße für Object Detection
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet Standards
        ]
    )


def get_validation_transforms():
    """
    Transform-Pipeline für Validation (keine Augmentierung)

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_test_transforms():
    """
    Transform-Pipeline für Test (identisch mit Validation)

    Returns:
        torchvision.transforms.Compose
    """
    return get_validation_transforms()
