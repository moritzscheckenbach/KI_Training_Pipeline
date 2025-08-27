import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes
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
    Erwartet YOLOv5/8-Style Labeldateien:
      <class_id> <x_center> <y_center> <width> <height>
    mit allen Koordinaten normiert auf [0,1].

    Gibt standardmäßig Boxes im Format XYWH (Pixel) zurück.
    Optionaler `img_id_start` sorgt für disjunkte image_ids je Split.
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        img_id_start: int = 0,  # wichtig für disjunkte IDs je Split
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_id_start = int(img_id_start)
        self.image_files = _list_images(images_dir)

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
                        # Ignoriere fehlerhafte Zeilen statt Exception zu werfen
                        continue
                    try:
                        class_id, x_c, y_c, w, h = map(float, parts[:5])
                    except ValueError:
                        continue

                    # YOLO (relativ, center) -> XYWH (Pixel, top-left)
                    bw = max(w * W, 0.0)
                    bh = max(h * H, 0.0)
                    x1 = (x_c * W) - bw / 2.0
                    y1 = (y_c * H) - bh / 2.0

                    # Clamp auf Canvas
                    x1 = max(min(x1, W - 1.0), 0.0)
                    y1 = max(min(y1, H - 1.0), 0.0)
                    # Breite/Höhe begrenzen, damit Box im Bild bleibt
                    bw = max(min(bw, W - x1), 0.0)
                    bh = max(min(bh, H - y1), 0.0)

                    if bw > 0 and bh > 0:
                        labels.append(int(class_id))
                        boxes.append([x1, y1, bw, bh])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.empty((0, 4), dtype=torch.float32)
            labels_t = torch.empty((0,), dtype=torch.int64)
        return boxes_t, labels_t

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        W, H = image.width, image.height

        boxes, labels = self._load_labels(label_path, W, H)

        target = {
            "boxes": boxes,                      # XYWH (Pixel)
            "labels": labels,                    # int64
            "image_id": torch.tensor(self.img_id_start + idx, dtype=torch.int64),
            "orig_size": torch.tensor([H, W], dtype=torch.int64),  # nützlich für Postprocessing
            "size": torch.tensor([H, W], dtype=torch.int64),
        }

        if self.transform is not None:
            # Bevorzugt: (img, target) -> (img, target)
            try:
                image, target = self.transform(image, target)
            except TypeError:
                image = self.transform(image)

        # tv_tensors -> plain Tensor (und auf gewünschtes Format bringen)
        # tv_tensors -> plain Tensor (und auf gewünschtes Format bringen)
        OUT_FMT = "xywh"  # falls dein Modell XYXY erwartet: "xyxy"

        def _fmt_to_str(fmt) -> str | None:
            if fmt is None:
                return None
            if isinstance(fmt, str):
                return fmt.lower()
            name = getattr(fmt, "name", None)
            if name:
                return name.lower()
            return str(fmt).lower()

        bb_obj = target.get("boxes")

        # Prüfe breit gefasst auf BoundingBoxes (kompatibel zu versch. torchvision-Versionen)
        if isinstance(bb_obj, TVBoundingBoxes) or "BoundingBoxes" in type(bb_obj).__name__:
            fmt_attr = getattr(bb_obj, "format", None)
            # Tensor extrahieren – versionssicher
            if hasattr(bb_obj, "as_tensor"):
                boxes_t = bb_obj.as_tensor()
            elif isinstance(bb_obj, torch.Tensor):
                boxes_t = bb_obj
            else:
                boxes_t = torch.as_tensor(bb_obj)
            boxes_t = boxes_t.to(torch.float32)

            in_fmt = _fmt_to_str(fmt_attr)
            if in_fmt is None or in_fmt == OUT_FMT:
                # Format unbekannt oder bereits korrekt -> unverändert lassen
                target["boxes"] = boxes_t
            else:
                target["boxes"] = box_convert(boxes_t, in_fmt=in_fmt, out_fmt=OUT_FMT)
        else:
            # Kein tv_tensor -> in float32 gießen
            b = target.get("boxes")
            if b is None:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            else:
                target["boxes"] = torch.as_tensor(b, dtype=torch.float32)

        if not isinstance(target.get("labels"), torch.Tensor):
            target["labels"] = torch.as_tensor(target.get("labels", []), dtype=torch.int64)

        return image, target
