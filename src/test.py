# train_frcnn_r50_fpn_coco.py
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

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
    evaluate_coco(model, val_loader, coco_val, contig_to_catid, device)


if __name__ == "__main__":
    main()
