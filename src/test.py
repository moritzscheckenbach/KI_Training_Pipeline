# train_frcnn_r50_fpn_coco.py
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------
# Utils
# -----------------------------
def collate_fn(batch):
    # Für torchvision detection models notwendig
    return tuple(zip(*batch))


def build_category_mappings(coco_gt: COCO) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Erzeugt Mapping von COCO-Category-IDs -> kontinuierliche IDs (1..K)
    sowie das inverse Mapping (kontinuierlich -> COCO-ID).
    Hintergrund: Torchvision-Detektionsmodelle erwarten Labels im Bereich [1..K].
    """
    cat_ids = sorted(coco_gt.getCatIds())
    catid_to_contig = {cid: i + 1 for i, cid in enumerate(cat_ids)}  # 0 ist "background" (implizit)
    contig_to_catid = {v: k for k, v in catid_to_contig.items()}
    return catid_to_contig, contig_to_catid


class CocoDetWrapped(CocoDetection):
    """
    Wrapper um torchvision.datasets.CocoDetection, der Targets in das
    erwartete Format der torchvision-Detektionsmodelle bringt:
      target = {
        "boxes": FloatTensor[N,4] in XYXY,
        "labels": Int64Tensor[N] (1..K),
        "image_id": Tensor([id]),
        "area": Tensor[N],
        "iscrowd": UInt8Tensor[N]
      }
    """

    def __init__(self, img_folder, ann_file, transforms, catid_to_contig: Dict[int, int]):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.catid_to_contig = catid_to_contig

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        # COCO liefert bbox als [x,y,w,h] und category_id als originale COCO-ID
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for a in anns:
            if "bbox" not in a:
                continue
            x, y, w, h = a["bbox"]
            # in XYXY konvertieren
            boxes.append([x, y, x + w, y + h])
            # Label auf kontinuierliche Skala mappen
            labels.append(self.catid_to_contig[a["category_id"]])
            area.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        if len(boxes) == 0:
            # Leere Platzhalter, damit die Tensors korrekt typisiert sind
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(area, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)

        image_id = self.ids[idx]
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def get_transforms(train: bool = True):
    # Minimal: nur ToTensor (keine Resize/Crop – FasterRCNN kann variable Größe)
    # Für robustes Training können augmentations ergänzt werden.
    if train:
        return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    else:
        return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def get_model(num_classes: int, pretrained: bool = True):
    """
    Faster R-CNN mit ResNet-50 + FPN.
    num_classes = Anzahl Kategorien + 1 (Hintergrund implizit, aber Head erwartet N Klassen inkl. Hintergrund-Index 0).
    Torchvision kapselt Hintergrund intern; der Classification-Head erwartet num_classes.
    """
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        # Head anpassen
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


@torch.no_grad()
def evaluate_coco(model, data_loader, coco_gt: COCO, contig_to_catid: Dict[int, int], device):
    """
    Läuft Inferenz auf dem Val-Loader, sammelt Detections im COCO-JSON-Format,
    wertet mit COCOeval aus und gibt die wichtigsten Kennzahlen aus.
    """
    model.eval()
    results: List[Dict[str, Any]] = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()

            # XYXY -> XYWH
            if boxes.numel() > 0:
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                w = x2 - x1
                h = y2 - y1
                xywh = torch.stack([x1, y1, w, h], dim=1)
                for b, s, l in zip(xywh, scores, labels):
                    results.append(
                        {
                            "image_id": img_id,
                            "category_id": int(contig_to_catid[int(l)]),
                            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "score": float(s),
                        }
                    )

    if len(results) == 0:
        print("WARN: Keine Predictions erzeugt – Evaluation wird übersprungen.")
        return

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # coco_eval.stats enthält u.a. AP@[.50:.95], AP50, AP75 usw.


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, scaler=None):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        # Warmup (optional, stabilisiert Start)
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        from torch.optim.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (i + 1) % print_freq == 0:
            loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            print(f"Epoch {epoch} [{i+1}/{len(data_loader)}]  total: {losses.item():.4f}  {loss_str}")


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN R50-FPN on COCO")
    parser.add_argument("--data_root", type=str, default="datasets/object_detection/Type_COCO/Test_Duckiebots", help="Pfad zum COCO-Root (enthält train/, valid/, test/)")
    parser.add_argument("--train_images", type=str, default="train")
    parser.add_argument("--val_images", type=str, default="valid")
    parser.add_argument("--train_ann", type=str, default="train/_annotations.coco.json")
    parser.add_argument("--val_ann", type=str, default="valid/_annotations.coco.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    # Pfad relativ zum aktuellen Arbeitsverzeichnis oder absolut setzen
    if not os.path.isabs(args.data_root):
        # Falls relativer Pfad, vom aktuellen Verzeichnis ausgehen
        data_root = Path.cwd() / args.data_root
    else:
        data_root = Path(args.data_root)
    
    train_imgs = data_root / args.train_images
    val_imgs = data_root / args.val_images
    train_ann = data_root / args.train_ann
    val_ann = data_root / args.val_ann

    print(f"Suche Datensatz in: {data_root}")
    print(f"Train-Bilder: {train_imgs}")
    print(f"Val-Bilder: {val_imgs}")
    print(f"Train-Annotation: {train_ann}")
    print(f"Val-Annotation: {val_ann}")

    assert train_imgs.is_dir(), f"Train-Bilder nicht gefunden: {train_imgs}"
    assert val_imgs.is_dir(), f"Val-Bilder nicht gefunden: {val_imgs}"
    assert train_ann.is_file(), f"Train-Annotation nicht gefunden: {train_ann}"
    assert val_ann.is_file(), f"Val-Annotation nicht gefunden: {val_ann}"

    # COCO Ground Truth laden (für Mappings & Evaluation)
    coco_train = COCO(str(train_ann))
    coco_val = COCO(str(val_ann))

    catid_to_contig, contig_to_catid = build_category_mappings(coco_train)
    num_classes = len(catid_to_contig) + 1  # +1 für Hintergrund

    # Datasets
    ds_train = CocoDetWrapped(
        img_folder=str(train_imgs),
        ann_file=str(train_ann),
        transforms=get_transforms(train=True),
        catid_to_contig=catid_to_contig,
    )
    ds_val = CocoDetWrapped(
        img_folder=str(val_imgs),
        ann_file=str(val_ann),
        transforms=get_transforms(train=False),
        catid_to_contig=catid_to_contig,  # identisches Mapping verwenden!
    )

    # DataLoader
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=1,  # für Evaluation oft 1 stabiler
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Modell
    model = get_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50, scaler=scaler)
        lr_scheduler.step()

    # -----------------------------
    # Evaluation (COCO mAP)
    # -----------------------------
    evaluate_coco_from_loader(model, val_loader, contig_to_catid, device, "bbox")


@torch.no_grad()
def _maybe_to_xywh_abs(bboxes_xywh: torch.Tensor,
                       img_w: int,
                       img_h: int) -> torch.Tensor:
    """
    Erwartet GT-Boxen im XYWH-Format. Falls normalisiert (<=1.5),
    skaliert auf Pixelkoordinaten.
    """
    if bboxes_xywh.numel() == 0:
        return bboxes_xywh
    # Heuristik: wenn alle Werte <= 1.5, behandeln als normalisiert.
    if float(bboxes_xywh.max()) <= 1.5:
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=bboxes_xywh.dtype, device=bboxes_xywh.device)
        return bboxes_xywh * scale
    return bboxes_xywh


def _get_img_wh_from_target(tgt: Dict[str, Any],
                            img_tensor: Optional[torch.Tensor]) -> (int, int):
    """
    Versucht width/height robust aus dem Target zu lesen.
    Fallback: Tensorgröße (C,H,W) falls verfügbar.
    """
    if "width" in tgt and "height" in tgt:
        return int(tgt["width"]), int(tgt["height"])
    if "orig_size" in tgt:
        # üblich: (H, W)
        h, w = tgt["orig_size"]
        return int(w), int(h)
    if "size" in tgt:
        h, w = tgt["size"]
        return int(w), int(h)
    if img_tensor is not None:
        # Tensor: (C, H, W)
        _, h, w = img_tensor.shape
        return int(w), int(h)
    # Letzter Fallback
    return 0, 0


@torch.no_grad()
def _build_coco_gt_from_loader(
    data_loader,
    contig_to_catid: Dict[int, int],
    device: torch.device
) -> COCO:
    """
    Erstellt ein COCO-GT-Objekt (in-memory) aus dem DataLoader.
    Erwartet Targets mit:
      - "image_id" (Tensor oder int),
      - "boxes" (XYWH; evtl. normalisiert),
      - "labels" (contiguous ids, die über contig_to_catid auf COCO cat_id gemappt werden),
      - optional: "width"/"height" oder "orig_size"/"size".
    """
    images = []
    annotations = []
    ann_id = 1  # laufende ID

    # categories aus Mapping ableiten (Namen optional)
    categories = [{"id": int(coco_id), "name": str(coco_id)} for coco_id in sorted(set(contig_to_catid.values()))]

    # Einmal durch den Loader iterieren (eval: shuffle=False)
    for batch in data_loader:
        # batch = (images, targets) oder nur targets – je nach collate_fn
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, tgts = batch
            # auf CPU lassen – wir brauchen hier nur Meta/Boxen
        else:
            # Unerwartetes Format
            raise ValueError("Unerwartetes Batch-Format. Erwartet (images, targets).")

        for img, tgt in zip(imgs, tgts):
            img_id = int(tgt["image_id"].item() if isinstance(tgt["image_id"], torch.Tensor) else tgt["image_id"])
            img_w, img_h = _get_img_wh_from_target(tgt, img)

            # Image-Eintrag
            images.append({
                "id": img_id,
                "width": img_w,
                "height": img_h,
                # "file_name" optional
            })

            # GT-Boxen lesen (im Dataset als XYXY) und in XYWH konvertieren
            gt_boxes = tgt["boxes"]
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.clone().cpu()
            else:
                gt_boxes = torch.as_tensor(gt_boxes)

            # Falls die Boxen normalisiert sind (selten), zuerst in Pixel skalieren
            if gt_boxes.numel() > 0 and float(gt_boxes.max()) <= 1.5 and img_w > 0 and img_h > 0:
                scale_xyxy = torch.tensor([img_w, img_h, img_w, img_h],
                                        dtype=gt_boxes.dtype, device=gt_boxes.device)
                gt_boxes = gt_boxes * scale_xyxy

            # XYXY -> XYWH
            if gt_boxes.numel() > 0:
                x1 = gt_boxes[:, 0]
                y1 = gt_boxes[:, 1]
                x2 = gt_boxes[:, 2]
                y2 = gt_boxes[:, 3]
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)
                gt_boxes = torch.stack([x1, y1, w, h], dim=1)



            gt_labels = tgt["labels"]
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu()

            for b, l in zip(gt_boxes, gt_labels):
                x, y, w, h = [float(v) for v in b.tolist()]
                coco_cat = int(contig_to_catid[int(l)])
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": coco_cat,
                    "bbox": [x, y, w, h],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                    "iscrowd": 0,
                })
                ann_id += 1

    # COCO-Objekt aus Dict erstellen
    coco = COCO()
    coco.dataset = {
        "info": {"description": "in-memory GT", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco.createIndex()
    return coco


@torch.no_grad()
def evaluate_coco_from_loader(
    model,
    data_loader,
    contig_to_catid: Dict[int, int],
    device: torch.device,
    iou_type: str = "bbox",
):
    """
    Führt Inferenz + COCOeval durch, ohne Annotationsdatei zu laden.
    GT kommt vollständig aus dem DataLoader.
    """
    # 1) COCO-GT aus Loader aufbauen
    coco_gt = _build_coco_gt_from_loader(data_loader, contig_to_catid, device)

    # 2) Inferenz & Predictions ins COCO-JSON-Format (XYWH, absolute Pixel)
    model.eval()
    results: List[Dict[str, Any]] = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item() if isinstance(tgt["image_id"], torch.Tensor) else tgt["image_id"])

            boxes_xyxy = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            if boxes_xyxy.numel() > 0:
                x1 = boxes_xyxy[:, 0]
                y1 = boxes_xyxy[:, 1]
                x2 = boxes_xyxy[:, 2]
                y2 = boxes_xyxy[:, 3]
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)
                xywh = torch.stack([x1, y1, w, h], dim=1)

                for b, s, l in zip(xywh, scores, labels):
                    results.append({
                        "image_id": img_id,
                        "category_id": int(contig_to_catid[int(l)]),
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(s),
                    })

    if len(results) == 0:
        print("WARN: Keine Predictions erzeugt – Evaluation wird übersprungen.")
        return None

    # 3) COCOeval
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Rückgabe der Stats (AP@[.50:.95], AP50, AP75, etc.)
    # coco_eval.stats: ndarray mit 12 Werten
    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }
    return metrics






















if __name__ == "__main__":
    main()
