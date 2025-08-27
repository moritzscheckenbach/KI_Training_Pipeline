import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes as TVBoundingBoxes

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _list_images(images_dir: str) -> List[str]:
    files = []
    for f in os.listdir(images_dir):
        p = os.path.join(images_dir, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in _VALID_EXTS:
            files.append(f)
    files.sort()
    return files

class YoloDataset(Dataset):
    """
    Erwartet YOLOv5/8-Style Labels:
      <class_id> <x_center> <y_center> <width> <height>  (alle normiert [0,1])
    Gibt (img, target) analog zum CocoDataset zurück, mit:
      target = {
        "boxes":   FloatTensor[N,4] im XYWH-Pixel-Format,
        "labels":  LongTensor[N],
        "image_id": LongTensor[1],
        "area":    FloatTensor[N],
        "iscrowd": LongTensor[N] (immer 0)
      }
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        img_id_start: int = 0,
        debug_mode: bool = False,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_id_start = int(img_id_start)
        self.image_files = _list_images(images_dir)
        self.debug_mode = bool(debug_mode)
        self.debug_samples = set(range(min(5, len(self.image_files)))) if self.debug_mode else set()

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_labels(self, label_path: str, W: int, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id, x_c, y_c, w, h = map(float, parts[:5])
                    except ValueError:
                        continue

                    bw = max(w * W, 0.0)
                    bh = max(h * H, 0.0)
                    x1 = (x_c * W) - bw / 2.0
                    y1 = (y_c * H) - bh / 2.0

                    # clamp ins Bild
                    x1 = max(min(x1, W - 1.0), 0.0)
                    y1 = max(min(y1, H - 1.0), 0.0)
                    bw = max(min(bw, W - x1), 0.0)
                    bh = max(min(bh, H - y1), 0.0)

                    if bw > 0 and bh > 0:
                        labels.append(int(class_id))
                        boxes.append([x1, y1, bw, bh])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        return boxes_t, labels_t

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        # Einheitlich RGB (wie im COCO-Wrapper)
        image = Image.open(img_path)
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        W, H = image.width, image.height

        boxes, labels = self._load_labels(label_path, W, H)

        # area & iscrowd wie bei COCO
        areas = (boxes[:, 2] * boxes[:, 3]) if boxes.numel() else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,                                    # XYWH float32
            "labels": labels,                                  # int64
            "image_id": torch.tensor(self.img_id_start + idx, dtype=torch.int64),
            "area": areas.to(torch.float32),
            "iscrowd": iscrowd,
            # optional, schadet nicht:
            "orig_size": torch.tensor([H, W], dtype=torch.int64),
            "size": torch.tensor([H, W], dtype=torch.int64),
        }

        # optionale Transforms (kompatibel zu (img, target)->(img, target) UND nur-img)
        if self.transform is not None:
            try:
                image, target = self.transform(image, target)
            except TypeError:
                image = self.transform(image)

        # Falls ein Transform tv_tensors.BoundingBoxes zurückliefert → Tensor + Format prüfen
        bb = target.get("boxes", None)
        if isinstance(bb, TVBoundingBoxes) or (bb is not None and "BoundingBoxes" in type(bb).__name__):
            fmt = getattr(bb, "format", None)
            boxes_t = bb.as_tensor() if hasattr(bb, "as_tensor") else torch.as_tensor(bb)
            boxes_t = boxes_t.to(torch.float32)
            # Falls Transform XYXY erzeugt hat, zurück nach XYWH
            in_fmt = getattr(fmt, "name", None)
            in_fmt = in_fmt.lower() if isinstance(in_fmt, str) else (fmt.lower() if isinstance(fmt, str) else None)
            if in_fmt == "xyxy":
                target["boxes"] = box_convert(boxes_t, in_fmt="xyxy", out_fmt="xywh")
            else:
                target["boxes"] = boxes_t
        else:
            target["boxes"] = torch.as_tensor(target.get("boxes", torch.zeros((0,4))), dtype=torch.float32)

        # Dtypes homogenisieren
        if not isinstance(target.get("labels"), torch.Tensor):
            target["labels"] = torch.as_tensor(target.get("labels", []), dtype=torch.int64)
        else:
            target["labels"] = target["labels"].to(torch.int64)

        if "area" in target:
            target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
        if "iscrowd" in target:
            target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        if "image_id" in target:
            target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int64)

        return image, target
