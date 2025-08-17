import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes

def xyxy_to_xywh(box_xyxy):
    # box_xyxy: [xmin, ymin, xmax, ymax]
    x_min, y_min, x_max, y_max = box_xyxy
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    return [x_min, y_min, w, h]

class PascalDataset(Dataset):
    """
    Liest Pascal VOC XML-Labels (XYXY in Pixeln) und gibt COCO-XYWH (Pixel) zurück.
    Erwartet:
      images_dir: JPG/PNG
      labels_dir: XML mit gleichem Basenamen
    class_to_id: dict wie {"person": 1, "car": 2, ...}
    """
    def __init__(self, images_dir, labels_dir, class_to_id, transform=None, clip_to_image=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.class_to_id = class_to_id
        self.clip_to_image = clip_to_image

        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.image_files.sort()  # stabile Reihenfolge

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        xml_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".xml")

        image = Image.open(img_path).convert("RGB")
        W, H = image.width, image.height

        boxes_xywh = []
        labels = []

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name is None:
                    continue
                # Optional: „difficult“ überspringen
                difficult = obj.findtext("difficult")
                if difficult is not None and difficult.strip() == "1":
                    continue

                bnd = obj.find("bndbox")
                if bnd is None:
                    continue

                # VOC ist 1-basiert in vielen Datasets; hier robust float-parsen
                def f(tag):
                    v = bnd.findtext(tag)
                    return float(v) if v is not None else None

                xmin, ymin, xmax, ymax = f("xmin"), f("ymin"), f("xmax"), f("ymax")
                if None in (xmin, ymin, xmax, ymax):
                    continue

                # In Pixel bleiben; optional clamping
                if self.clip_to_image:
                    xmin = max(0.0, min(xmin, W - 1.0))
                    ymin = max(0.0, min(ymin, H - 1.0))
                    xmax = max(0.0, min(xmax, W - 1.0))
                    ymax = max(0.0, min(ymax, H - 1.0))

                if xmax <= xmin or ymax <= ymin:
                    continue

                box_xywh = xyxy_to_xywh([xmin, ymin, xmax, ymax])
                boxes_xywh.append(box_xywh)

                # Klassen-ID aus Mapping
                if name not in self.class_to_id:
                    # Unbekannte Klassen optional überspringen
                    continue
                labels.append(int(self.class_to_id[name]))

        boxes = torch.tensor(boxes_xywh, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        # Transforms (v2 bevorzugt: (img, target))
        if self.transform:
            try:
                image, target = self.transform(image, target)
            except TypeError:
                image = self.transform(image)

        # tv_tensors → plain Tensor in COCO-XYWH
        if isinstance(target.get("boxes"), BoundingBoxes):
            target["boxes"] = target["boxes"].to_format("XYWH").as_tensor()
        if not isinstance(target.get("boxes"), torch.Tensor):
            target["boxes"] = torch.as_tensor(target.get("boxes", []), dtype=torch.float32)
        if not isinstance(target.get("labels"), torch.Tensor):
            target["labels"] = torch.as_tensor(target.get("labels", []), dtype=torch.int64)

        return image, target
