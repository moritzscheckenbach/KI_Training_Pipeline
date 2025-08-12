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
from torchvision.datasets import ImageFolder


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    """
    Training pipeline für Klassifizierung (ImageNet-Ordnerstruktur).
    Logs und Struktur wie Original, aber statt Objekt­erkennung einfache Bildklassifizierung.
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
    logger.add(lambda msg: print(msg, end=""),
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
               level="INFO", colorize=True)

    log_file_path = f"{experiment_dir}/training.log"
    logger.add(log_file_path,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
               level="DEBUG", rotation="100 MB", retention="10 days")

    shutil.copy("conf/config.yaml", f"{experiment_dir}/configs/config.yaml")

    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    logger.info(f"📝 Log file: {log_file_path}")

    # =============================================================================
    # 3. MODEL LOADING
    # =============================================================================
    try:
        model_architecture = importlib.import_module(f"model_architecture.{model_type}.{model_name}")
        model = model_architecture.build_model(num_classes=cfg.dataset.num_classes)
        if transfer_learning_enabled:
            logger.info("✅ Transfer learning model loaded successfully")
        else:
            logger.info("✅ Fresh model loaded successfully")
    except ImportError as e:
        logger.error(f"❌ Error loading model architecture: {e}")
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

    transform_train = transforms.Compose([
        transforms.Resize((inputsize_x, inputsize_y)),
        base_transform,
        transforms.ToTensor()
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize((inputsize_x, inputsize_y)),
        transforms.ToTensor()
    ])

    # =============================================================================
    # 5. DATASET LOADING (ImageFolder)
    # =============================================================================
    logger.info(f"📊 Loading {dataset_type} dataset from: {dataset_root}")

    if dataset_type == "Type_Folder":
        train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=transform_train)
        val_dataset = ImageFolder(root=os.path.join(dataset_root, "valid"), transform=transform_val_test)
        test_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=transform_val_test)
    else:
        raise ValueError(f"❌ Unsupported dataset type for classification: {dataset_type}")

    logger.info(f"📈 Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # =============================================================================
    # 6. MODEL TO DEVICE
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clear_gpu_cache()
    logger.info(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")
        logger.info(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # =============================================================================
    # 7. LOSS & OPTIMIZER
    # =============================================================================
    criterion = nn.CrossEntropyLoss()

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
    # 9. TENSORBOARD
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

        # ---- Training ----
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = 100.0 * correct_train / total_train

        # ---- Validation ----
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = 100.0 * correct_val / total_val

        logger.info(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Scheduler step
        if use_scheduler:
            if cfg.scheduler.type == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{experiment_dir}/models/best_model_weights.pth")
            torch.save({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{experiment_dir}/models/best_model_info.pth")
            logger.info(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.warning("⏹️ Early stopping triggered.")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": cfg
        }, f"{experiment_dir}/models/last_model.pth")

    # =============================================================================
    # 11. TEST EVALUATION
    # =============================================================================
    logger.info("🧪 Starting test evaluation")
    model.load_state_dict(torch.load(f"{experiment_dir}/models/best_model_weights.pth", weights_only=True))

    model.eval()
    test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    avg_test_loss = test_loss / len(test_dataloader)
    test_acc = 100.0 * correct_test / total_test

    logger.info(f"🧪 Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    writer.add_scalar("Test/Loss", avg_test_loss, cfg.training.epochs)
    writer.add_scalar("Test/Accuracy", test_acc, cfg.training.epochs)

    # =============================================================================
    # 12. EXPERIMENT SUMMARY
    # =============================================================================
    summary = {
        "experiment_name": experiment_name,
        "model_architecture": model_name,
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "test_accuracy": test_acc,
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


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
