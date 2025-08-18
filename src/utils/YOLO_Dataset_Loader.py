import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        self.image_files.sort()  # WICHTIG: Reihenfolge stabil halten

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        W, H = image.width, image.height

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # YOLO: normalized center-format -> convert to XYWH (top-left) in pixels
                    bw = width * W
                    bh = height * H
                    x1 = (x_center * W) - bw / 2.0
                    y1 = (y_center * H) - bh / 2.0
                    labels.append(int(class_id))
                    boxes.append([x1, y1, bw, bh])  # XYWH in pixels

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transform:
            # Prefer (img, target) signature (v2 wrapper). Fallback to image-only if needed.
            try:
                image, target = self.transform(image, target)
            except TypeError:
                image = self.transform(image)

        # Ensure plain tensors for downstream code
        if isinstance(target.get("boxes"), BoundingBoxes):
            target["boxes"] = target["boxes"].to_format("XYWH").as_tensor()
        if not isinstance(target.get("boxes"), torch.Tensor):
            target["boxes"] = torch.as_tensor(target.get("boxes", []), dtype=torch.float32)
        if not isinstance(target.get("labels"), torch.Tensor):
            target["labels"] = torch.as_tensor(target.get("labels", []), dtype=torch.int64)

        return image, target
