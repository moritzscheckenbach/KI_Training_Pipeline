import gc
import importlib
import os
import shutil
from datetime import datetime
from collections import defaultdict

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import yaml
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection

from utils.YOLO_Dataset_Loader import YoloDataset


def calculate_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Sehr vereinfachte Berechnung von Accuracy, Precision, Recall, mAP.
    - pred_boxes, gt_boxes: List[Tensor] pro Bild
    - pred_labels, gt_labels: List[Tensor] pro Bild
    """
    tp = 0
    fp = 0
    fn = 0
    total = 0
    
    for pb, pl, gb, gl in zip(pred_boxes, pred_labels, gt_boxes, gt_labels):
        total += len(gl)
        matched = set()
        for i, pbox in enumerate(pb):
            ious = []
            for j, gbox in enumerate(gb):
                # IoU berechnen
                ixmin = max(pbox[0], gbox[0])
                iymin = max(pbox[1], gbox[1])
                ixmax = min(pbox[0]+pbox[2], gbox[0]+gbox[2])
                iymax = min(pbox[1]+pbox[3], gbox[1]+gbox[3])
                iw = max(0, ixmax - ixmin)
                ih = max(0, iymax - iymin)
                inter = iw * ih
                union = pbox[2]*pbox[3] + gbox[2]*gbox[3] - inter
                iou = inter / union if union > 0 else 0
                ious.append((iou, j))
            # Max IoU wählen
            if ious:
                best_iou, best_j = max(ious, key=lambda x: x[0])
                if best_iou >= iou_threshold and pl[i] == gl[best_j] and best_j not in matched:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
        fn += len(gb) - len(matched)
    
    accuracy = tp / total if total > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    mAP = precision  # stark vereinfacht, hier ≈ Precision
    
    return accuracy, precision, recall, mAP


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    """
    Main function for training the model.
    This function sets up the training environment, loads the model, datasets, and starts the training loop.


    Example of using variables from the config:
    random_seed = cfg.training.random_seed

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
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO", colorize=True)

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

    # =============================================================================
    # 5. DATASET LOADING
    # =============================================================================

    logger.info(f"📊 Loading {dataset_type} dataset from: {dataset_root}")

    # TYPE COCO Dataset
    if dataset_type == "Type_COCO":
        train_dataset = CocoDetection(
            root=f"{dataset_root}train/",
            annFile=f"{dataset_root}train/_annotations.coco.json",
            transform=transforms.Compose([transforms.Resize((inputsize_x, inputsize_y)), base_transform, transforms.ToTensor()]),
        )
        val_dataset = CocoDetection(
            root=f"{dataset_root}valid/", annFile=f"{dataset_root}valid/_annotations.coco.json", transform=transforms.Compose([transforms.Resize((inputsize_x, inputsize_y)), transforms.ToTensor()])
        )
        test_dataset = CocoDetection(
            root=f"{dataset_root}test/", annFile=f"{dataset_root}test/_annotations.coco.json", transform=transforms.Compose([transforms.Resize((inputsize_x, inputsize_y)), transforms.ToTensor()])
        )

    # TYPE YOLO Dataset
    if dataset_type == "Type_YOLO":
        transform_train = transforms.Compose([transforms.Resize((inputsize_x, inputsize_y)), base_transform, transforms.ToTensor()])
        transform_val_test = transforms.Compose([transforms.Resize((inputsize_x, inputsize_y)), transforms.ToTensor()])
        train_dataset = YoloDataset(images_dir=f"{dataset_root}train/images/", labels_dir=f"{dataset_root}train/labels/", transform=transform_train)
        val_dataset = YoloDataset(images_dir=f"{dataset_root}valid/images/", labels_dir=f"{dataset_root}valid/labels/", transform=transform_val_test)
        test_dataset = YoloDataset(images_dir=f"{dataset_root}test/images/", labels_dir=f"{dataset_root}test/labels/", transform=transform_val_test)

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
    # 10. TRAINING LOOP
    # =============================================================================

    best_val_loss = float("inf")
    patience_counter = 0
    log_interval = 20  # alle 20 Batches loggen

    logger.info(f"🎯 Starting training loop for {cfg.training.epochs} epochs")
    logger.info(f"⚙️ Batch size: {cfg.training.batch_size}, Learning rate: {cfg.training.learning_rate}")
    logger.info(f"⏰ Early stopping patience: {cfg.training.early_stopping_patience}")

    for epoch in range(cfg.training.epochs):
        logger.info(f"📈 Epoch {epoch+1}/{cfg.training.epochs}")

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        batch_loss_components = defaultdict(float)

        # Variables für Image-Logging
        first_batch_logged = False

        for i, (images, targets) in enumerate(train_dataloader):
            if images is None or targets is None:
                continue

            images = torch.stack([image.to(device) for image in images])

            # --- Image Logging (nur erste Batch pro Epoch) ---
            if not first_batch_logged and (epoch + 1) % 5 == 0:  # Alle 5 Epochen
                # Normalisiere Bilder für bessere Darstellung (0-1 Range)
                display_images = images[:min(8, len(images))].clone()  # Max 8 Bilder
                
                # Falls Bilder normalisiert wurden, denormalisieren
                # Annahme: ImageNet Normalisierung wurde verwendet
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                display_images = display_images * std + mean
                display_images = torch.clamp(display_images, 0, 1)
                
                img_grid = torchvision.utils.make_grid(display_images, nrow=4, padding=2, normalize=False)
                writer.add_image('Training_Samples', img_grid, global_step=epoch)
                first_batch_logged = True
                logger.debug(f"   📸 Training images logged for epoch {epoch+1}")

            processed_targets = []
            for target_list in targets:
                if isinstance(target_list, list) and len(target_list) > 0:
                    target_dict = {
                        "boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in target_list]),
                        "labels": torch.tensor([ann["category_id"] for ann in target_list])
                    }
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:
                    processed_targets.append({
                        "boxes": torch.empty((0, 4), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device)
                    })

            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            
            # Loss-Komponenten für Batch-Logging sammeln
            for k, v in loss_dict.items():
                batch_loss_components[k] += v.item() if hasattr(v, 'item') else v

            # --- Batch Logging ---
            if (i + 1) % log_interval == 0:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar("Loss/Train_batch", losses.item(), step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f"Loss_Components/Train_batch/{k}", v.item() if hasattr(v, 'item') else v, step)
                logger.debug(f"   Batch {i+1}/{len(train_dataloader)}, Loss: {losses.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)
        # Durchschnittliche Loss-Komponenten für Epoch
        for k in batch_loss_components:
            batch_loss_components[k] /= len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_loss_components = defaultdict(float)
        all_preds, all_pred_labels, all_gts, all_gt_labels = [], [], [], []

        # Variable für Validation Image-Logging
        val_images_logged = False

        with torch.no_grad():
            for images, targets in val_dataloader:
                if images is None or targets is None:
                    continue

                images = torch.stack([image.to(device) for image in images])

                # --- Validation Image Logging (nur erste Batch pro Epoch) ---
                if not val_images_logged and (epoch + 1) % 5 == 0:  # Alle 5 Epochen
                    display_images = images[:min(8, len(images))].clone()
                    
                    # Denormalisierung (falls nötig)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                    display_images = display_images * std + mean
                    display_images = torch.clamp(display_images, 0, 1)
                    
                    img_grid = torchvision.utils.make_grid(display_images, nrow=4, padding=2, normalize=False)
                    writer.add_image('Validation_Samples', img_grid, global_step=epoch)
                    val_images_logged = True
                    logger.debug(f"   📸 Validation images logged for epoch {epoch+1}")

                processed_targets = []
                for target_list in targets:
                    if isinstance(target_list, list) and len(target_list) > 0:
                        target_dict = {
                            "boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in target_list]),
                            "labels": torch.tensor([ann["category_id"] for ann in target_list])
                        }
                        processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                    else:
                        processed_targets.append({
                            "boxes": torch.empty((0, 4), device=device),
                            "labels": torch.empty((0,), dtype=torch.long, device=device)
                        })

                loss_dict = model(images, processed_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Validation Loss-Komponenten sammeln
                for k, v in loss_dict.items():
                    val_loss_components[k] += v.item() if hasattr(v, 'item') else v

                # Dummy Prediction für Metrics (hier sollten echte Predictions verwendet werden)
                preds = [{"boxes": torch.rand((len(t["labels"]), 4))*100, "labels": t["labels"]} for t in processed_targets]
                all_preds.extend([p["boxes"].cpu() for p in preds])
                all_pred_labels.extend([p["labels"].cpu() for p in preds])
                all_gts.extend([t["boxes"].cpu() for t in processed_targets])
                all_gt_labels.extend([t["labels"].cpu() for t in processed_targets])

        avg_val_loss = val_loss / len(val_dataloader)
        # Durchschnittliche Validation Loss-Komponenten
        for k in val_loss_components:
            val_loss_components[k] /= len(val_dataloader)

        # Metriken berechnen
        acc, prec, rec, map_score = calculate_metrics(all_preds, all_pred_labels, all_gts, all_gt_labels)

        logger.info(f"   Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logger.info(f"   Metrics - Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, mAP: {map_score:.3f}")

        # --- Scheduler Step ---
        if use_scheduler:
            if cfg.scheduler.type == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            logger.debug(f"   Current Learning Rate: {current_lr:.6f}")

        # --- Model Parameter Histograms (alle 10 Epochen) ---
        if (epoch + 1) % 10 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param, global_step=epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step=epoch)
            logger.debug(f"   📊 Model histograms logged for epoch {epoch+1}")

        # --- Epoch TensorBoard Logging ---
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        # Loss-Komponenten loggen
        for k, v in batch_loss_components.items():
            writer.add_scalar(f"Loss_Components/Train/{k}", v, epoch)
        for k, v in val_loss_components.items():
            writer.add_scalar(f"Loss_Components/Validation/{k}", v, epoch)

        # Metriken loggen
        writer.add_scalar("Metrics/Validation/Accuracy", acc, epoch)
        writer.add_scalar("Metrics/Validation/Precision", prec, epoch)
        writer.add_scalar("Metrics/Validation/Recall", rec, epoch)
        writer.add_scalar("Metrics/Validation/mAP", map_score, epoch)

        # Speichere auch letztes Model
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_train_loss, "val_loss": avg_val_loss, "config": cfg},
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
                "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "mAP": map_score}
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
    # 11. TEST EVALUATION
    # =============================================================================

    logger.info("🧪 Starting test evaluation")
    model.load_state_dict(torch.load(f"{experiment_dir}/models/best_model_weights.pth", weights_only=True))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in test_dataloader:
            if images is None or targets is None:
                continue
            images = torch.stack([image.to(device) for image in images])
            processed_targets = []
            for target_list in targets:
                if isinstance(target_list, list) and len(target_list) > 0:  # ← Füge "and len(target_list) > 0" hinzu
                    target_dict = {"boxes": torch.stack([torch.tensor(ann["bbox"]) for ann in target_list]), "labels": torch.tensor([ann["category_id"] for ann in target_list])}
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:  # ← Füge diesen else-Block hinzu
                    # Leere targets
                    processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()
    avg_test_loss = test_loss / len(test_dataloader)
    logger.info(f"🧪 Test Loss: {avg_test_loss:.4f}")
    writer.add_scalar("Test/Loss", avg_test_loss, cfg.training.epochs)

    # =============================================================================
    # 12. EXPERIMENT SUMMARY
    # =============================================================================

    # Konvertiere OmegaConf zu normalem Dictionary für saubere YAML-Ausgabe
    from omegaconf import OmegaConf

    clean_config = OmegaConf.to_container(cfg, resolve=True)

    summary = {
        "experiment_name": experiment_name,
        "model_architecture": model_name,
        "timestamp": timestamp,
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "config": clean_config,  # Saubere Konfiguration ohne OmegaConf-Metadaten
    }

    with open(f"{experiment_dir}/experiment_summary.yaml", "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
    writer.close()

    # =============================================================================
    # 13. FINAL LOGGING
    # =============================================================================

    logger.success(f"✅ Training completed!")
    logger.info(f"📁 Results saved in: {experiment_dir}")
    logger.info(f"🏆 Best model: {experiment_dir}/models/best_model.pth")
    logger.info(f"📊 TensorBoard: tensorboard --logdir={experiment_dir}/tensorboard")
    logger.info(f"📋 Summary: {experiment_dir}/experiment_summary.yaml")
    logger.info(f"📝 Training log: {log_file_path}")


def collate_fn(batch):
    """Füllt leere Annotations mit Nullen statt sie zu filtern"""
    images = []
    targets = []

    for img, target in batch:
        if img is None:
            continue

        images.append(img)

        # Statt filtern: leere targets beibehalten
        if target is None or len(target) == 0:
            targets.append([])
        else:
            targets.append(target)

    if len(images) == 0:
        return None, None

    # Behalte das gleiche Format wie vorher
    return (tuple(images), tuple(targets))


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
