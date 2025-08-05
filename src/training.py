import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Modular imports - diese sollten als separate Module implementiert werden
from datasets.object_detection.coco_dataset import COCODataset
from augmentations.object_detection_augmentation import get_augmentation_transforms
from model_architecture.object_detection.model_factory import get_model
from utils.optimizer_factory import get_optimizer, get_scheduler
from utils.checkpoint_utils import save_checkpoint, load_checkpoint


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device, epoch: int, config: Dict[str, Any]) -> float:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()

        if config["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        optimizer.step()

        total_loss += losses.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        if config["log_wandb"] and batch_idx % config["log_interval"] == 0:
            wandb.log({"train_loss": losses.item(), "epoch": epoch, "batch": batch_idx})

    return total_loss / num_batches


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / num_batches


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler, epoch: int, loss: float, config: Dict[str, Any], filepath: str):
    """Save model checkpoint"""
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss, "config": config}

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: optim.Optimizer, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration"""
    return {
        # Data paths
        "train_data_path": "/path/to/coco/train2017",
        "train_annotation_file": "/path/to/coco/annotations/instances_train2017.json",
        "val_data_path": "/path/to/coco/val2017",
        "val_annotation_file": "/path/to/coco/annotations/instances_val2017.json",
        # Model parameters
        "model_name": "fasterrcnn_resnet50",
        "num_classes": 91,  # COCO has 80 classes + background
        "pretrained": True,
        "trainable_backbone_layers": 3,
        # Training parameters
        "num_epochs": 50,
        "batch_size": 4,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "gradient_clipping": True,
        "max_grad_norm": 10.0,
        # Optimizer
        "optimizer": "SGD",  # SGD, Adam, AdamW
        # Scheduler
        "scheduler": "StepLR",  # StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
        "step_size": 7,
        "gamma": 0.1,
        "milestones": [16, 22],
        "T_max": 50,
        "eta_min": 0,
        "reduce_mode": "min",
        "reduce_factor": 0.1,
        "reduce_patience": 10,
        # Data augmentation
        "image_size": 800,
        "use_horizontal_flip": True,
        "horizontal_flip_prob": 0.5,
        "use_vertical_flip": False,
        "vertical_flip_prob": 0.5,
        "use_rotation": True,
        "rotation_limit": 15,
        "rotation_prob": 0.3,
        "use_blur": True,
        "blur_limit": 3,
        "blur_prob": 0.2,
        "use_brightness_contrast": True,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "brightness_contrast_prob": 0.3,
        "use_hue_saturation": True,
        "hue_shift_limit": 20,
        "sat_shift_limit": 30,
        "val_shift_limit": 20,
        "hue_saturation_prob": 0.3,
        "use_gaussian_noise": True,
        "gaussian_noise_var": (10.0, 50.0),
        "gaussian_noise_prob": 0.2,
        "use_cutout": False,
        "cutout_num_holes": 8,
        "cutout_max_h_size": 8,
        "cutout_max_w_size": 8,
        "cutout_prob": 0.2,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        # Hardware
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "pin_memory": True,
        # Logging and saving
        "log_wandb": False,
        "wandb_project": "object-detection",
        "wandb_entity": None,
        "log_interval": 100,
        "save_interval": 5,
        "output_dir": "./checkpoints",
        "resume_from_checkpoint": None,
        # Validation
        "validate_every": 1,
        "early_stopping": True,
        "early_stopping_patience": 15,
        "save_best_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Train object detection model on COCO dataset")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--train_data_path", type=str, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")

    args = parser.parse_args()

    # Load configuration
    config = get_default_config()

    if args.config:
        with open(args.config, "r") as f:
            config.update(json.load(f))

    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup wandb logging
    if config["log_wandb"]:
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    # Setup device
    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")

    # Create datasets
    train_transforms = get_augmentation_transforms(config)
    val_transforms = A.Compose(
        [A.Resize(config["image_size"], config["image_size"]), A.Normalize(mean=config["normalize_mean"], std=config["normalize_std"]), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    train_dataset = COCODataset(config["train_data_path"], config["train_annotation_file"], transforms=train_transforms)

    val_dataset = COCODataset(config["val_data_path"], config["val_annotation_file"], transforms=val_transforms)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=config["pin_memory"], collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"], collate_fn=collate_fn)

    # Initialize model
    model = get_model(config)
    model.to(device)

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if config["resume_from_checkpoint"]:
        start_epoch, _ = load_checkpoint(config["resume_from_checkpoint"], model, optimizer, scheduler)
        logger.info(f"Resumed training from epoch {start_epoch}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config["num_epochs"]):
        # Train
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, config)

        # Validate
        if epoch % config["validate_every"] == 0:
            val_loss = validate(model, val_dataloader, device)
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if config["log_wandb"]:
                wandb.log({"epoch": epoch, "train_loss_epoch": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if not config["save_best_only"] or is_best or epoch % config["save_interval"] == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, checkpoint_path)

                if is_best:
                    best_checkpoint_path = output_dir / "best_checkpoint.pth"
                    save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, best_checkpoint_path)

            # Early stopping
            if config["early_stopping"] and patience_counter >= config["early_stopping_patience"]:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

        # Update learning rate
        if scheduler:
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

    logger.info("Training completed!")

    # Save final model
    final_checkpoint_path = output_dir / "final_checkpoint.pth"
    save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, final_checkpoint_path)

    if config["log_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
