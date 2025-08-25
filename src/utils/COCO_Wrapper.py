import torch
from torchvision.ops import box_convert
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes


class COCOWrapper:
    """Wrappt image+Target für v2, konvertiert COCO-Listen/Dicts nach BoundingBoxes und zurück (XYWH),
    ohne Zusatz-Keys (z. B. image_id) zu verlieren.
    """

    def __init__(self, base_tf, inputsize_x, inputsize_y, debug_enabled=False):
        # HINWEIS: Resize übernimmt hier unverändert (inputsize_x, inputsize_y),
        # wie in deinem Originalcode.
        self.debug_enabled = debug_enabled
        self.base_tf = T.Compose(
            [
                base_tf,
                T.Resize((inputsize_y, inputsize_x)),
            ]
        )

    def __call__(self, img, target):
        # Canvas-Size bestimmen (H, W)
        if hasattr(img, "size"):  # PIL: (W, H)
            w, h = img.size
        elif hasattr(img, "shape") and len(img.shape) == 3:  # Tensor: CxHxW
            _, h, w = img.shape
        else:
            h = w = None  # wird von BoundingBoxes für einige Ops benötigt

        # Original-Target flach kopieren, um Keys nachher ggf. zu restaurieren
        original_target = target if isinstance(target, dict) else None

        # --- Eingaben normalisieren -> Dict mit BoundingBoxes im Format XYWH ---
        if isinstance(target, list):
            # Liste aus COCO-Annots -> Minimales Dict (XYWH + labels)
            boxes, labels = [], []
            for ann in target:
                bbox = ann.get("bbox")
                if bbox is None:
                    continue
                x, y, bw, bh = bbox  # COCO: XYWH
                boxes.append([x, y, bw, bh])
                labels.append(ann.get("category_id", 0))

            boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

            target = {
                "boxes": BoundingBoxes(boxes_t, format="XYWH", canvas_size=(h, w)),
                "labels": labels_t,
            }

        elif isinstance(target, dict):
            # Dict: ALLE Keys erhalten, nur boxes/labels sauber in tv_tensors packen
            tgt = dict(target)  # flache Kopie
            boxes = tgt.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            labels = tgt.get("labels", torch.zeros((0,), dtype=torch.int64))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            tgt["boxes"] = BoundingBoxes(boxes, format="XYWH", canvas_size=(h, w))
            tgt["labels"] = labels
            target = tgt

        else:
            raise TypeError(f"COCOWrapper: unerwarteter target-type {type(target)}")

        # --- Transforms anwenden ---
        img, target = self.base_tf(img, target)

        # --- BoundingBoxes zurück in normalen Tensor (XYWH) ---
        if isinstance(target, dict):
            bb = target.get("boxes", None)
            if isinstance(bb, BoundingBoxes):
                boxes_t = torch.as_tensor(bb, dtype=torch.float32)
                # Falls ein Transform das Format geändert hat, zurück auf XYWH bringen
                fmt = getattr(bb, "format", None)
                fmt_str = fmt.lower() if isinstance(fmt, str) else None
                if fmt_str and fmt_str != "xywh":
                    boxes_t = box_convert(boxes_t, in_fmt=fmt_str, out_fmt="xywh")
                target["boxes"] = boxes_t
            elif bb is not None:
                target["boxes"] = torch.as_tensor(bb, dtype=torch.float32)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            # --- Zusatz-Keys restaurieren, falls während der Transforms verloren ---
            if original_target is not None and isinstance(original_target, dict):
                for k, v in original_target.items():
                    if k not in target:
                        target[k] = v

            # Dtypes defensiv angleichen (ohne Logik/Geometrie zu ändern)
            if "labels" in target:
                target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
            if "iscrowd" in target:
                target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
            if "area" in target and torch.is_tensor(target["area"]):
                target["area"] = target["area"].to(torch.float32)
            if "image_id" in target:
                target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int64)

        return img, target
