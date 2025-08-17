import gc
import importlib
import os
import shutil
from datetime import datetime
from math import sqrt

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes

from utils.YOLO_Dataset_Loader import YoloDataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    """
    Main function for training the model.
    This function sets up the training environment, loads the model, datasets, and starts the training loop.
    """

    # =============================================================================
    # 1. EXPERIMENT SETUP
    # =============================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_root = cfg.dataset.root
    dataset_type = cfg.dataset.type

    transfer_learning_enabled = cfg.model.transfer_learning.enabled
    model_type = cfg.model.type

    if transfer_learning_enabled:
        model_name = cfg.model.transfer_learning.trans_file
        logger.info("🔄 Using Transfer Learning Mode")
        experiment_name = f"{timestamp}_{cfg.model.file}_transfer"
    else:
        model_name = cfg.model.file
        experiment_name = f"{timestamp}_{model_name}"
        logger.info("🆕 Using Fresh Training Mode")

    experiment_dir = f"trained_models/{model_type}/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/tensorboard", exist_ok=True)
    os.makedirs(f"{experiment_dir}/configs", exist_ok=True)

    # =============================================================================
    # 2. LOGGER SETUP
    # =============================================================================

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="DEBUG", colorize=True)

    log_file_path = f"{experiment_dir}/training.log"
    logger.add(log_file_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG", rotation="100 MB", retention="10 days")

    src = to_absolute_path("conf/config.yaml")
    shutil.copy(src, os.path.join(experiment_dir, "configs", "config.yaml"))

    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    logger.info(f"📝 Log file: {log_file_path}")

    # =============================================================================
    # 3. MODEL LOADING
    # =============================================================================

    try:
        if cfg.model.transfer_learning.enabled:
            model_architecture = importlib.import_module(f"model_architecture.{model_type}.{model_name}")
            model = model_architecture.build_model_tr(cfg=cfg)
            logger.info("✅ Transfer learning model loaded successfully")
        else:
            model_architecture = importlib.import_module(f"model_architecture.{model_type}.{model_name}")
            model = model_architecture.build_model(num_classes=cfg.dataset.num_classes)
            logger.info("✅ Fresh model loaded successfully")
    except ImportError as e:
        logger.error(f"❌ Error loading model architecture: {e}")
        logger.error(f"Looking for: model_architecture.{model_type}.{model_name}")
        raise
    except Exception as e:
        logger.error(f"❌ Error building model: {e}")
        raise

    # =============================================================================
    # 4. AUGMENTATION
    # =============================================================================

    inputsize_x, inputsize_y = model.get_input_size()
    logger.info(f"📐 Model input size: {inputsize_x}x{inputsize_y}")

    augmentation = importlib.import_module(f"augmentations.{cfg.augmentation.file}")
    base_transform = augmentation.augment()
    logger.info(f"🔄 Using augmentation: {cfg.augmentation.file}")

    # v2 Eval-Transform (deterministisch)
    val_base_transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((inputsize_x, inputsize_y)),
            T.SanitizeBoundingBoxes(),
        ]
    )

    v2_train_tf = DetectionV2Wrapper(base_transform, inputsize_x, inputsize_y)
    v2_eval_tf = DetectionV2Wrapper(val_base_transform, inputsize_x, inputsize_y)

    # =============================================================================
    # 5. DATASET LOADING
    # =============================================================================

    logger.info(f"📊 Loading {dataset_type} dataset from: {dataset_root}")

    # TYPE COCO Dataset
    if dataset_type == "Type_COCO":
        train_dataset = CocoDetection(
            root=f"{dataset_root}train/",
            annFile=f"{dataset_root}train/_annotations.coco.json",
            transforms=v2_train_tf,
        )
        val_dataset = CocoDetection(
            root=f"{dataset_root}valid/",
            annFile=f"{dataset_root}valid/_annotations.coco.json",
            transforms=v2_eval_tf,
        )
        test_dataset = CocoDetection(
            root=f"{dataset_root}test/",
            annFile=f"{dataset_root}test/_annotations.coco.json",
            transforms=v2_eval_tf,
        )

    # TYPE YOLO Dataset
    if dataset_type == "Type_YOLO":
        train_dataset = YoloDataset(images_dir=f"{dataset_root}train/images/", labels_dir=f"{dataset_root}train/labels/", transform=v2_train_tf)
        val_dataset = YoloDataset(images_dir=f"{dataset_root}valid/images/", labels_dir=f"{dataset_root}valid/labels/", transform=v2_eval_tf)
        test_dataset = YoloDataset(images_dir=f"{dataset_root}test/images/", labels_dir=f"{dataset_root}test/labels/", transform=v2_eval_tf)

    logger.info(f"📈 Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create PyTorch DataLoaders for each split
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # =============================================================================
    # 6. MODEL TO DEVICE
    # =============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        clear_gpu_cache()
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")
        logger.info(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # =============================================================================
    # 7. OPTIMIZER SETUP
    # =============================================================================

    if cfg.model.transfer_learning.enabled and "backbone_lr_multiplier" in cfg.model.transfer_learning.lr:
        backbone_lr = cfg.training.learning_rate * cfg.model.transfer_learning.lr.backbone_lr_multiplier
        head_lr = cfg.training.learning_rate * cfg.model.transfer_learning.lr.head_lr_multiplier
        param_groups = []
        # Add backbone parameters
        if hasattr(model.base_model, "backbone"):
            param_groups.append({"params": model.base_model.backbone.parameters(), "lr": backbone_lr, "name": "backbone"})
        # Add detection head parameters
        if hasattr(model.base_model, "detection_head"):
            param_groups.append({"params": model.base_model.detection_head.parameters(), "lr": head_lr, "name": "detection_head"})
        # Add any other parameters
        other_params = []
        backbone_params = set(model.base_model.backbone.parameters()) if hasattr(model.base_model, "backbone") else set()
        head_params = set(model.base_model.detection_head.parameters()) if hasattr(model.base_model, "detection_head") else set()
        for param in model.parameters():
            if param not in backbone_params and param not in head_params:
                other_params.append(param)
        if other_params:
            param_groups.append({"params": other_params, "lr": cfg.training.learning_rate, "name": "other"})
        optimizer_lib = importlib.import_module("utils.optimizer")
        optimizer = optimizer_lib.get_optimizer(param_groups, cfg)
        logger.info(f"🔧 Using Transfer Learning Optimizer:")
        logger.info(f"   Backbone LR: {backbone_lr}")
        logger.info(f"   Head LR: {head_lr}")
    else:
        optimizer_lib = importlib.import_module("utils.optimizer")
        optimizer = optimizer_lib.get_optimizer(model.parameters(), cfg)
        logger.info(f"🔧 Using optimizer: {cfg.optimizer.type}")

    # =============================================================================
    # 8. SCHEDULER SETUP
    # =============================================================================

    scheduler_module = cfg.scheduler.file
    if scheduler_module:
        scheduler_lib = importlib.import_module(f"utils.{scheduler_module}")
        scheduler = scheduler_lib.get_scheduler(optimizer, cfg)
        use_scheduler = True
        logger.info(f"📅 Using scheduler: {cfg.scheduler.type}")
    else:
        scheduler = None
        use_scheduler = False
        logger.info("📅 No scheduler configured")

    # =============================================================================
    # 9. TENSORBOARD SETUP
    # =============================================================================

    writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
    logger.info(f"📊 TensorBoard logs: {experiment_dir}/tensorboard")

    # =============================================================================
    # 10. TRAINING LOOP (mit erweitertem Logging)
    # =============================================================================

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"🎯 Starting training loop for {cfg.training.epochs} epochs")
    logger.info(f"⚙️ Batch size: {cfg.training.batch_size}, Learning rate: {cfg.training.learning_rate}")
    logger.info(f"⏰ Early stopping patience: {cfg.training.early_stopping_patience}")

    global_step = 0
    batch_log_interval = getattr(cfg, "batch_log_interval", 20)
    log_images_every_n_epochs = getattr(cfg, "log_images_every_n_epochs", 2)

    # Einmalige Logs
    n_params = sum(p.numel() for p in model.parameters())
    writer.add_scalar("Model/num_parameters", n_params, 0)

    for epoch in range(cfg.training.epochs):
        logger.info(f"📈 Epoch {epoch+1}/{cfg.training.epochs}")

        # ------------------------
        # Training Phase
        # ------------------------
        model.train()
        train_loss = 0.0

        for i, (images, targets) in enumerate(train_dataloader):
            if images is None or targets is None:
                continue

            images = torch.stack([image.to(device) for image in images])

            processed_targets = []
            for tgt in targets:
                if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                    boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                    labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                    processed_targets.append({"boxes": boxes.to(device).float(), "labels": labels.to(device).long()})
                elif isinstance(tgt, list) and len(tgt) > 0:
                    target_dict = {"boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in tgt]), "labels": torch.tensor([ann["category_id"] for ann in tgt])}
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:
                    processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})

            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            # Grad-Histogramme (optional: nur jede n-te Epoche)
            if epoch % 1 == 0 and i == (len(train_dataloader) - 1):
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        writer.add_histogram(f"Grads/{name}", p.grad.detach().cpu().numpy(), epoch)

            optimizer.step()

            train_loss += losses.item()

            # --- Batch-Level TensorBoard Logging ---
            global_step += 1
            writer.add_scalar("Batch/Loss", losses.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Batch/LossComponents/{k}", (v.item() if isinstance(v, torch.Tensor) else float(v)), global_step)

            for gi, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"LR/group_{gi}", pg["lr"], global_step)

            try:
                gnorm = grad_global_norm(model.parameters())
                writer.add_scalar("Grad/global_norm", gnorm, global_step)
            except Exception:
                pass

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                writer.add_scalar("GPU/mem_alloc_MB", mem_alloc, global_step)
                writer.add_scalar("GPU/mem_reserved_MB", mem_reserved, global_step)

            if (i + 1) % batch_log_interval == 0:
                logger.debug(f"   Batch {i+1}/{len(train_dataloader)}, Loss: {losses.item():.4f}, GradNorm: {gnorm if 'gnorm' in locals() else 'n/a'}")

            # --- Train Image-Logging (erste Batch, alle n Epochen) ---
            if i == 0 and (epoch % log_images_every_n_epochs == 0):
                with torch.no_grad():
                    model.eval()
                    imgs_vis = images[:4].detach().cpu()
                    preds = model(imgs_vis.to(device))
                    model.train()
                grid = make_gt_vs_pred_grid(imgs_vis, processed_targets[: len(imgs_vis)], preds)
                writer.add_image("Train/GT_vs_Pred", grid, epoch)

        avg_train_loss = train_loss / len(train_dataloader)

        # Parameter-Histogramme (einmal pro Epoche)
        for name, p in model.named_parameters():
            writer.add_histogram(f"Params/{name}", p.detach().cpu().numpy(), epoch)

        # ------------------------
        # Validation Phase
        # ------------------------
        # Wichtig: Für Loss von Torchvision-Detektoren train()+no_grad verwenden
        val_loss = 0.0
        model.train()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_dataloader):
                if images is None or targets is None:
                    continue
                images = torch.stack([image.to(device) for image in images])

                processed_targets = []
                for tgt in targets:
                    if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                        boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                        labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                        processed_targets.append({"boxes": boxes.to(device).float(), "labels": labels.to(device).long()})
                    elif isinstance(tgt, list) and len(tgt) > 0:
                        target_dict = {"boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in tgt]), "labels": torch.tensor([ann["category_id"] for ann in tgt])}
                        processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                    else:
                        processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})

                # Val-Loss
                loss_dict = model(images, processed_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                # Val-Bilder (erste Batch): echte Detections in eval()
                if batch_idx == 0 and (epoch % log_images_every_n_epochs == 0):
                    model.eval()
                    with torch.no_grad():
                        imgs_vis = images[:4].detach().cpu()
                        preds = model(imgs_vis.to(device))
                    model.train()
                    grid = make_gt_vs_pred_grid(imgs_vis, processed_targets[: len(imgs_vis)], preds)
                    writer.add_image("Val/GT_vs_Pred", grid, epoch)

        avg_val_loss = val_loss / len(val_dataloader)
        logger.info(f"   Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, targets in val_dataloader:
                if images is None or targets is None:
                    continue
                images = torch.stack([image.to(device) for image in images])

                # GT in xyxy + auf Device (Klassencodierung: category_id muss 0..C-1 sein)
                processed_targets = []
                for tgt in targets:
                    if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                        boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                        labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                        t = {"boxes": coco_xywh_to_xyxy(boxes.to(device).float()), "labels": labels.to(device).long()}
                    elif isinstance(tgt, list) and len(tgt) > 0:
                        t = {
                            "boxes": coco_xywh_to_xyxy(torch.stack([torch.tensor(ann["bbox"]) for ann in tgt]).to(device)),
                            "labels": torch.tensor([ann["category_id"] for ann in tgt], device=device, dtype=torch.long),
                        }
                    else:
                        t = {"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)}
                    processed_targets.append(t)

                detections = model(images)  # list of dicts (eval)

                for det, gt in zip(detections, processed_targets):
                    pred_item = {
                        "boxes": det.get("boxes", torch.empty((0, 4), device=device)).detach().cpu(),
                        "scores": det.get("scores", torch.empty((0,), device=device)).detach().cpu(),
                        "labels": det.get("labels", torch.empty((0,), dtype=torch.long, device=device)).detach().cpu(),
                    }
                    gt_item = {
                        "boxes": gt["boxes"].detach().cpu(),
                        "labels": gt["labels"].detach().cpu(),
                    }
                    all_preds.append(pred_item)
                    all_gts.append(gt_item)

        num_classes = cfg.dataset.num_classes
        cm_ext_val = confusion_matrix_detection(all_preds, all_gts, num_classes=num_classes, iou_thr=0.5, score_thr=0.5)
        m = metrics_from_cm(cm_ext_val)

        # --- TensorBoard Logging (Epoche) ---
        writer.add_scalar("ValMetrics/Accuracy", m["accuracy"], epoch)
        writer.add_scalar("ValMetrics/Micro/Precision", m["micro"]["precision"], epoch)
        writer.add_scalar("ValMetrics/Micro/Recall", m["micro"]["recall"], epoch)
        writer.add_scalar("ValMetrics/Micro/F1", m["micro"]["f1"], epoch)
        writer.add_scalar("ValMetrics/Macro/Precision", m["macro"]["precision"], epoch)
        writer.add_scalar("ValMetrics/Macro/Recall", m["macro"]["recall"], epoch)
        writer.add_scalar("ValMetrics/Macro/F1", m["macro"]["f1"], epoch)

        # Per-Class als Histogramme/Scalars
        pc = m["per_class"]
        writer.add_histogram("ValMetrics/PerClass/Precision", pc["precision"], epoch)
        writer.add_histogram("ValMetrics/PerClass/Recall", pc["recall"], epoch)
        writer.add_histogram("ValMetrics/PerClass/F1", pc["f1"], epoch)

        # --- Scheduler Step ---
        if use_scheduler:
            if cfg.scheduler.type == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            logger.debug(f"   Current Learning Rate: {current_lr:.6f}")

        # --- Epoch-Level Scalars ---
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Einzelne Loss-Komponenten (von letzter Val-Iteration)
        if isinstance(loss_dict, dict):
            for loss_name, loss_value in loss_dict.items():
                scalar_value = loss_value.item() if isinstance(loss_value, torch.Tensor) else float(loss_value)
                writer.add_scalar(f"Loss_Components/{loss_name}", scalar_value, epoch)

        # Speichere auch letztes Model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": cfg,
            },
            f"{experiment_dir}/models/last_model.pth",
        )

        # Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), f"{experiment_dir}/models/best_model_weights.pth")

            checkpoint_info = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_info, f"{experiment_dir}/models/best_model_info.pth")

            logger.info(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.debug(f"   Patience counter: {patience_counter}/{cfg.training.early_stopping_patience}")
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.warning("⏹️ Early stopping triggered.")
                break

    # =============================================================================
    # 11. TEST EVALUATION (Loss)
    # =============================================================================

    logger.info("🧪 Starting test evaluation (loss)")
    model.load_state_dict(torch.load(f"{experiment_dir}/models/best_model_weights.pth", weights_only=True))
    model.train()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in test_dataloader:
            if images is None or targets is None:
                continue
            images = torch.stack([image.to(device) for image in images])
            processed_targets = []
            for tgt in targets:
                if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                    boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                    labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                    processed_targets.append({"boxes": boxes.to(device).float(), "labels": labels.to(device).long()})
                elif isinstance(tgt, list) and len(tgt) > 0:
                    target_dict = {"boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in tgt]), "labels": torch.tensor([ann["category_id"] for ann in tgt])}
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:
                    processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()
    avg_test_loss = test_loss / len(test_dataloader)
    logger.info(f"🧪 Test Loss: {avg_test_loss:.4f}")
    writer.add_scalar("Test/Loss", avg_test_loss, cfg.training.epochs)

    # =============================================================================
    # 12. CONFUSION MATRIX AUF TESTDATEN (Detections + CM in TensorBoard)
    # =============================================================================

    logger.info("🧪 Building confusion matrix on test set")
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in test_dataloader:
            if images is None or targets is None:
                continue
            images = torch.stack([image.to(device) for image in images])

            processed_targets = []
            for tgt in targets:
                if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                    boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                    labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                    t = {"boxes": coco_xywh_to_xyxy(boxes.to(device).float()), "labels": labels.to(device).long()}
                elif isinstance(tgt, list) and len(tgt) > 0:
                    t = {
                        "boxes": coco_xywh_to_xyxy(torch.stack([torch.tensor(ann["bbox"]) for ann in tgt]).to(device)),
                        "labels": torch.tensor([ann["category_id"] for ann in tgt], device=device, dtype=torch.long),
                    }
                else:
                    t = {"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)}
                processed_targets.append(t)

            detections = model(images)  # eval -> list of dicts

            for det, gt in zip(detections, processed_targets):
                pred_item = {
                    "boxes": det.get("boxes", torch.empty((0, 4), device=device)).detach().cpu(),
                    "scores": det.get("scores", torch.empty((0,), device=device)).detach().cpu(),
                    "labels": det.get("labels", torch.empty((0,), dtype=torch.long, device=device)).detach().cpu(),
                }
                gt_item = {
                    "boxes": gt["boxes"].detach().cpu(),
                    "labels": gt["labels"].detach().cpu(),
                }
                all_preds.append(pred_item)
                all_gts.append(gt_item)

    num_classes = cfg.dataset.num_classes
    cm_ext = confusion_matrix_detection(all_preds, all_gts, num_classes=num_classes, iou_thr=0.5, score_thr=0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_ext, interpolation="nearest")
    ax.set_title("Confusion Matrix (GT rows, Pred cols)\nLast column=FN, Last row=FP")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth class")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_classes + 1))
    ax.set_xticklabels([str(i) for i in range(num_classes)] + ["FP"])
    ax.set_yticklabels([str(i) for i in range(num_classes)] + ["FN"])
    plt.tight_layout()
    writer.add_figure("Test/ConfusionMatrix_IOU0.5", fig, global_step=0)
    plt.close(fig)

    # =============================================================================
    # 13. EXPERIMENT SUMMARY
    # =============================================================================

    clean_config = OmegaConf.to_container(cfg, resolve=True)
    summary = {
        "experiment_name": experiment_name,
        "model_architecture": model_name,
        "timestamp": timestamp,
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "config": clean_config,
    }

    with open(f"{experiment_dir}/experiment_summary.yaml", "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
    writer.close()

    # =============================================================================
    # 14. FINAL LOGGING
    # =============================================================================

    logger.success(f"✅ Training completed!")
    logger.info(f"📁 Results saved in: {experiment_dir}")
    logger.info(f"🏆 Best model: {experiment_dir}/models/best_model.pth")
    logger.info(f"📊 TensorBoard: tensorboard --logdir={experiment_dir}/tensorboard")
    logger.info(f"📋 Summary: {experiment_dir}/experiment_summary.yaml")
    logger.info(f"📝 Training log: {log_file_path}")


# ==============================
# Hilfsfunktionen
# ==============================


def collate_fn(batch):
    """Detection-collate: gebe Listen zurück, kompatibel mit variabler Boxanzahl."""
    images = []
    targets = []
    for img, target in batch:
        if img is None:
            continue
        images.append(img)
        if target is None or (isinstance(target, list) and len(target) == 0):
            targets.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
        else:
            targets.append(target)
    if len(images) == 0:
        return None, None
    return list(images), list(targets)


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def coco_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] in [x,y,w,h] -> [x1,y1,x2,y2]
    if boxes.numel() == 0:
        return boxes
    xyxy = boxes.clone()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy


def tensor_to_uint8(img: torch.Tensor) -> torch.Tensor:
    # erwartet CHW float [0,1] oder [0,255]
    x = img.detach().cpu()
    if x.max() <= 1.0:
        x = x * 255.0
    return x.to(torch.uint8)


def draw_boxes_on_img(img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # img: CHW uint8, boxes: xyxy
    if boxes.numel() == 0:
        return draw_bounding_boxes(img, torch.zeros((0, 4), dtype=torch.int64), labels=[], width=2)
    return draw_bounding_boxes(img, boxes.round().to(torch.int64), labels=[str(int(l)) for l in labels], width=2)


def make_gt_vs_pred_grid(imgs_vis: torch.Tensor, targets_list, preds_list):
    """
    imgs_vis: Tensor [B, C, H, W] on CPU
    targets_list: list of dicts with boxes (xyxy on device) + labels
    preds_list: list of dicts with boxes + labels (xyxy)
    """
    panels = []
    for b_idx in range(len(imgs_vis)):
        img_u8 = tensor_to_uint8(imgs_vis[b_idx])
        # GT
        gt_t = targets_list[b_idx]
        gt_boxes_xyxy = coco_xywh_to_xyxy(gt_t["boxes"].detach().cpu())
        gt_labels = gt_t["labels"].detach().cpu()
        gt_img = draw_boxes_on_img(img_u8, gt_boxes_xyxy, gt_labels)

        # Preds
        p = preds_list[b_idx]
        p_boxes = p.get("boxes", torch.empty((0, 4))).detach().cpu()
        p_labels = p.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
        pred_img = draw_boxes_on_img(img_u8, p_boxes, p_labels)

        panel = torch.cat([gt_img, pred_img], dim=2)  # nebeneinander
        panels.append(panel)

    grid = torchvision.utils.make_grid(panels, nrow=1)
    return grid


def grad_global_norm(parameters) -> float:
    tot_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            g = p.grad.detach()
            tot_sq += float(g.norm(2).item() ** 2)
    return sqrt(tot_sq)


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [Na,4],[Nb,4] in xyxy
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    lt = torch.max(a[:, None, :2], b[:, :2])  # [Na,Nb,2]
    rb = torch.min(a[:, None, 2:], b[:, 2:])  # [Na,Nb,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b - inter
    return inter / union.clamp(min=1e-9)


def confusion_matrix_detection(preds, gts, num_classes: int, iou_thr: float = 0.5, score_thr: float = 0.5) -> np.ndarray:
    """
    Einfache CM für Detection:
    - Greedy 1:1 Matching per IOU
    - Matrix: GT x Pred; rechte Randspalte=FN je GT-Klasse; untere Randzeile=FP je Pred-Klasse
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    fn_per_class = np.zeros(num_classes, dtype=np.int64)
    fp_per_class = np.zeros(num_classes, dtype=np.int64)

    for pred, gt in zip(preds, gts):
        p_boxes = pred["boxes"]
        p_scores = pred.get("scores", torch.ones((p_boxes.shape[0],)))
        p_labels = pred["labels"]
        keep = p_scores >= score_thr
        p_boxes = p_boxes[keep]
        p_labels = p_labels[keep]

        g_boxes = gt["boxes"]
        g_labels = gt["labels"]

        if p_boxes.numel() == 0 and g_boxes.numel() == 0:
            continue

        ious = iou_matrix(g_boxes, p_boxes)  # [Ng,Np]

        matched_g = set()
        matched_p = set()

        pairs = []
        for gi in range(ious.shape[0]):
            for pi in range(ious.shape[1]):
                if ious[gi, pi] >= iou_thr:
                    pairs.append((float(ious[gi, pi].item()), gi, pi))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, gi, pi in pairs:
            if gi in matched_g or pi in matched_p:
                continue
            gt_cls = int(g_labels[gi].item())
            pr_cls = int(p_labels[pi].item())
            cm[gt_cls, pr_cls] += 1
            matched_g.add(gi)
            matched_p.add(pi)

        for gi in range(len(g_labels)):
            if gi not in matched_g:
                fn_per_class[int(g_labels[gi].item())] += 1

        for pi in range(len(p_labels)):
            if pi not in matched_p:
                fp_per_class[int(p_labels[pi].item())] += 1

    cm_ext = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    cm_ext[:num_classes, :num_classes] = cm
    cm_ext[:num_classes, -1] = fn_per_class
    cm_ext[-1, :num_classes] = fp_per_class
    return cm_ext


def metrics_from_cm(cm_ext: np.ndarray):
    """
    Erwartet erweiterte CM (GT-Zeilen, Pred-Spalten, letzte Spalte=FN, letzte Zeile=FP).
    Liefert Mikro-/Makro-Precision/Recall/F1, Accuracy sowie per-class Werte.
    """
    num_classes = cm_ext.shape[0] - 1
    cm = cm_ext[:num_classes, :num_classes]
    fn = cm_ext[:num_classes, -1]
    fp = cm_ext[-1, :num_classes]

    tp = np.diag(cm)
    support = tp + fn  # GT je Klasse
    predicted = tp + fp  # Pred je Klasse

    with np.errstate(divide="ignore", invalid="ignore"):
        prec_c = np.where(predicted > 0, tp / predicted, 0.0)
        rec_c = np.where(support > 0, tp / support, 0.0)
        f1_c = np.where((prec_c + rec_c) > 0, 2 * prec_c * rec_c / (prec_c + rec_c), 0.0)

    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    precision_micro = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall_micro = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    valid = support > 0
    if valid.any():
        precision_macro = float(np.mean(prec_c[valid]))
        recall_macro = float(np.mean(rec_c[valid]))
        f1_macro = float(np.mean(f1_c[valid]))
    else:
        precision_macro = recall_macro = f1_macro = 0.0

    total_gt = support.sum()
    accuracy = float(TP / total_gt) if total_gt > 0 else 0.0

    return {
        "per_class": {"precision": prec_c, "recall": rec_c, "f1": f1_c, "support": support},
        "micro": {"precision": precision_micro, "recall": recall_micro, "f1": f1_micro},
        "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro},
        "accuracy": accuracy,
    }

class DetectionV2Wrapper:
    """Wrappt Bild+Target für v2, konvertiert COCO-Listen/Dicts nach BoundingBoxes und zurück (XYWH)."""

    def __init__(self, base_tf, inputsize_x, inputsize_y):
        # Ergänze Resize auf Modell-Inputgröße vor die eigentliche Augmentierung
        self.base_tf = T.Compose(
            [
                T.Resize((inputsize_x, inputsize_y)),
                base_tf,
            ]
        )

    def __call__(self, img, target):
        # Canvas-Size (H,W)
        if hasattr(img, "size"):
            w, h = img.size  # PIL: (W,H)
        elif hasattr(img, "shape") and len(img.shape) == 3:
            _, h, w = img.shape
        else:
            h = w = None

        if isinstance(target, list):
            boxes = []
            labels = []
            for ann in target:
                bbox = ann.get("bbox")
                cid = ann.get("category_id", 0)
                if bbox is None:
                    continue
                x, y, bw, bh = bbox  # COCO XYWH (Pixel)
                boxes.append([x, y, bw, bh])
                labels.append(cid)
            boxes_t = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
            target = {"boxes": BoundingBoxes(boxes_t, format="XYWH", canvas_size=(h, w)), "labels": labels_t}
        elif isinstance(target, dict):
            boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
            boxes = BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYWH", canvas_size=(h, w))
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

        img, target = self.base_tf(img, target)
        # Zurück zu Standard-Tensoren (XYWH) ohne tv_tensors.
        bb = target.get("boxes") if isinstance(target, dict) else None
        if isinstance(bb, BoundingBoxes):
            boxes_t = torch.as_tensor(bb, dtype=torch.float32)
            fmt = getattr(bb, "format", None)
            fmt_str = fmt.lower() if isinstance(fmt, str) else None
            if fmt_str and fmt_str != "xywh":
                from torchvision.ops import box_convert

                boxes_t = box_convert(boxes_t, in_fmt=fmt_str, out_fmt="xywh")
            target["boxes"] = boxes_t
        return img, target



if __name__ == "__main__":
    train()
