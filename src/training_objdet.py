import gc
import importlib
import os
import shutil
from datetime import datetime

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection

from utils.YOLO_Dataset_Loader import YoloDataset


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

    shutil.copy("conf/config.yaml", f"{experiment_dir}/configs/config.yaml")

    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    logger.info(f"📝 Log file: {log_file_path}")

    # =============================================================================
    # 3. MODEL LOADING
    # =============================================================================

    try:
        if cfg.model.transfer_learning.enabled:
            model_architecture = importlib.import_module(f"model_architecture.{model_type}.{model_name}")

            model = model_architecture.build_model(num_classes=cfg.dataset.num_classes)  # TODO num_classes / pretrained / other arguments need to be handled
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
    clear_gpu_cache()
    logger.info(f"🖥️ Using device: {device}")
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

    logger.info(f"🎯 Starting training loop for {cfg.training.epochs} epochs")
    logger.info(f"⚙️ Batch size: {cfg.training.batch_size}, Learning rate: {cfg.training.learning_rate}")
    logger.info(f"⏰ Early stopping patience: {cfg.training.early_stopping_patience}")

    for epoch in range(cfg.training.epochs):
        logger.info(f"📈 Epoch {epoch+1}/{cfg.training.epochs}")

        # Training Phase
        model.train()
        train_loss = 0.0

        for i, (images, targets) in enumerate(train_dataloader):
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

            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss += losses.item()

            # Log every 100 batches
            if (i + 1) % 100 == 0:
                logger.debug(f"   Batch {i+1}/{len(train_dataloader)}, Loss: {losses.item():.4f}")

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation Phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_dataloader:
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
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_dataloader)

        logger.info(f"   Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Scheduler Step ---
        if use_scheduler:
            if cfg.scheduler.type == "ReduceLROnPlateau":
                # ReduceLROnPlateau braucht die Metric (val_loss)
                scheduler.step(avg_val_loss)
            else:
                # Andere Scheduler brauchen nur step()
                scheduler.step()

            # Log current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            logger.debug(f"   Current Learning Rate: {current_lr:.6f}")

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Einzelne Loss-Komponenten
        if isinstance(loss_dict, dict):
            for loss_name, loss_value in loss_dict.items():
                # Convert to float regardless of type
                if isinstance(loss_value, torch.Tensor):
                    scalar_value = loss_value.item()
                else:
                    scalar_value = float(loss_value)
                writer.add_scalar(f"Loss_Components/{loss_name}", scalar_value, epoch)

        # Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save model state dict separately (safer)
            torch.save(model.state_dict(), f"{experiment_dir}/models/best_model_weights.pth")

            # Save other data as separate file
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

        # Speichere auch letztes Model
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_train_loss, "val_loss": avg_val_loss, "config": cfg},
            f"{experiment_dir}/models/last_model.pth",
        )

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

    summary = {
        "experiment_name": experiment_name,
        "model_architecture": model_name,
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "config": cfg,
        "timestamp": timestamp,
    }

    with open(f"{experiment_dir}/experiment_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
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
