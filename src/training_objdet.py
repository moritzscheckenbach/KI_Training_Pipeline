import gc
import importlib
import os
import shutil
from datetime import datetime
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import OmegaConf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from conf.config import AIPipelineConfig
from utils.COCO_Loader import CocoDataset
from utils.PASCAL_Loader import PascalDataset
from utils.YOLO_Loader import YoloDataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: AIPipelineConfig):
    """
    Main function for training the model.
    This function sets up the training environment, loads the model, datasets, and starts the training loop.
    """

    # =============================================================================
    # EXPERIMENT SETUP
    # =============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_root = cfg.dataset.root
    dataset_type = cfg.dataset.type

    transfer_learning_enabled = cfg.model.transfer_learning.enabled
    model_type = cfg.model.type

    if transfer_learning_enabled:
        model_name = cfg.model.transfer_learning.trans_file
        experiment_name = f"{timestamp}_{cfg.model.file}_transfer"
    else:
        model_name = cfg.model.file
        experiment_name = f"{timestamp}_{model_name}"

    experiment_dir = f"trained_models/{model_type}/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/tensorboard", exist_ok=True)
    os.makedirs(f"{experiment_dir}/configs", exist_ok=True)

    # =============================================================================
    # LOGGER SETUP
    # =============================================================================
    debug_mode = cfg.training.debug_mode
    if debug_mode:
        logger_level = "DEBUG"
    else:
        logger_level = "INFO"

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level=logger_level, colorize=True)

    log_file_path = f"{experiment_dir}/training.log"
    logger.add(log_file_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level=logger_level, rotation="100 MB", retention="10 days")

    src = to_absolute_path("conf/config.yaml")
    shutil.copy(src, os.path.join(experiment_dir, "configs", "config.yaml"))

    if debug_mode:
        logger.info("🔍 Debug mode is ENABLED - will check for black images and provide detailed logs")

    if transfer_learning_enabled:
        logger.info("🔄 Using Transfer Learning Mode")
    else:
        logger.info("🆕 Using Fresh Training Mode")
    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📁 Experiment directory: {experiment_dir}")
    logger.info(f"📝 Log file: {log_file_path}")

    # =============================================================================
    # MODEL LOADING
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

    # # ---- TESTING MODEL INFERENCE MODE ----
    # logger.info("Testing model inference mode...")
    # model.eval()
    # dummy_input = torch.randn(2, 3, 416, 416).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # with torch.no_grad():
    #     test_output = model(dummy_input)
    # logger.info(f"Test output type: {type(test_output)}")
    # if isinstance(test_output, list):
    #     logger.info(f"Test output length: {len(test_output)}")
    #     if len(test_output) > 0:
    #         logger.info(f"First element keys: {list(test_output[0].keys()) if isinstance(test_output[0], dict) else 'not a dict'}")
    # model.train()
    # return

    # =============================================================================
    # AUGMENTATION
    # =============================================================================
    if cfg.model.transfer_learning.enabled:
        inputsize_x, inputsize_y = model_architecture.get_input_size(cfg=cfg)
    else:
        inputsize_x, inputsize_y = model_architecture.get_input_size()
    logger.info(f"📐 Model input size: {inputsize_x}x{inputsize_y}")

    augmentation = importlib.import_module(f"augmentations.{cfg.augmentation.file}")
    base_transform = augmentation.augment()
    logger.info(f"🔄 Using augmentation: {cfg.augmentation.file}")

    # v2 Eval-Transform (deterministisch)
    val_base_transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.SanitizeBoundingBoxes(),
        ]
    )

    v2_train_tf = COCOWrapper(base_transform, inputsize_x, inputsize_y)
    v2_eval_tf = COCOWrapper(val_base_transform, inputsize_x, inputsize_y)

    # =============================================================================
    # DATASET LOADING
    # =============================================================================
    logger.info(f"📊 Loading {dataset_type} dataset from: {dataset_root}")

    # TYPE COCO Dataset
    if dataset_type == "Type_COCO":
        train_dataset = CocoDataset(
            root=f"{dataset_root}train/",
            annFile=f"{dataset_root}train/_annotations.coco.json",
            transforms=v2_train_tf,
            img_id_start=1000000,
            debug_mode=debug_mode,
        )
        val_dataset = CocoDataset(
            root=f"{dataset_root}valid/",
            annFile=f"{dataset_root}valid/_annotations.coco.json",
            transforms=v2_eval_tf,
            img_id_start=2000000,
            debug_mode=debug_mode,
        )
        test_dataset = CocoDataset(
            root=f"{dataset_root}test/",
            annFile=f"{dataset_root}test/_annotations.coco.json",
            transforms=v2_eval_tf,
            img_id_start=3000000,
            debug_mode=debug_mode,
        )

    # TYPE YOLO Dataset
    if dataset_type == "Type_YOLO":
        train_dataset = YoloDataset(images_dir=f"{dataset_root}train/images/", labels_dir=f"{dataset_root}train/labels/", transform=v2_train_tf)
        val_dataset = YoloDataset(images_dir=f"{dataset_root}valid/images/", labels_dir=f"{dataset_root}valid/labels/", transform=v2_eval_tf)
        test_dataset = YoloDataset(images_dir=f"{dataset_root}test/images/", labels_dir=f"{dataset_root}test/labels/", transform=v2_eval_tf)

    # TYPE Pascal Dataset
    if dataset_type == "Type_Pascal_V10":
        train_dataset = PascalDataset(images_dir=f"{dataset_root}train/images/", labels_dir=f"{dataset_root}train/labels/", transform=v2_train_tf)
        val_dataset = PascalDataset(images_dir=f"{dataset_root}valid/images/", labels_dir=f"{dataset_root}valid/labels/", transform=v2_eval_tf)
        test_dataset = PascalDataset(images_dir=f"{dataset_root}test/images/", labels_dir=f"{dataset_root}test/labels/", transform=v2_eval_tf)

    logger.info(f"📈 Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create PyTorch DataLoaders for each split in COCO Format
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    if debug_mode:
        for i, (images, targets) in enumerate(train_dataloader):
            for n in range(len(images)):
                img = images[n]
                _check_img_range(img=img, img_id=f"dateloader_test img={n}")

    # =============================================================================
    # MODEL TO DEVICE
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        clear_gpu_cache()
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")
        logger.info(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # =============================================================================
    # OPTIMIZER SETUP
    # =============================================================================
    if cfg.model.transfer_learning.enabled and "backbone_lr_multiplier" in cfg.model.transfer_learning.lr:
        backbone_lr = cfg.training.learning_rate * cfg.model.transfer_learning.lr.backbone_lr_multiplier
        head_lr = cfg.training.learning_rate * cfg.model.transfer_learning.lr.head_lr_multiplier
        param_groups = []

        # Backbone dynamisch aus Config
        backbone_name = getattr(cfg.model.transfer_learning, "backbone_name", None)
        backbone_params = set()
        if backbone_name and hasattr(model.base_model, backbone_name):
            backbone = getattr(model.base_model, backbone_name)
            param_groups.append({"params": backbone.parameters(), "lr": backbone_lr, "name": backbone_name})
            backbone_params = set(backbone.parameters())

        # Head dynamisch aus Config
        head_name = getattr(cfg.model.transfer_learning, "head_name", None)
        head_params = set()
        if head_name and hasattr(model.base_model, head_name):
            head = getattr(model.base_model, head_name)
            param_groups.append({"params": head.parameters(), "lr": head_lr, "name": head_name})
            head_params = set(head.parameters())

        # Alle anderen Parameter (nicht Backbone, nicht Head)
        other_params = []
        for param in model.parameters():
            if param not in backbone_params and param not in head_params:
                other_params.append(param)
        if other_params:
            param_groups.append({"params": other_params, "lr": cfg.training.learning_rate, "name": "other"})

        optimizer_lib = importlib.import_module("utils.optimizer")
        optimizer = optimizer_lib.get_optimizer(param_groups, cfg)

        logger.info("🔧 Using Transfer Learning Optimizer:")
        logger.info(f"   {backbone_name} LR: {backbone_lr}")
        logger.info(f"   {head_name} LR: {head_lr}")

    else:
        optimizer_lib = importlib.import_module("utils.optimizer")
        optimizer = optimizer_lib.get_optimizer(model.parameters(), cfg)
        logger.info(f"🔧 Using optimizer: {cfg.optimizer.type}")

    # =============================================================================
    # SCHEDULER SETUP
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
    # TENSORBOARD SETUP
    # =============================================================================
    writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
    logger.info(f"📊 TensorBoard logs: {experiment_dir}/tensorboard")

    # =============================================================================
    # TRAINING LOOP (mit erweitertem Logging)
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
        clear_gpu_cache()
        logger.info(f"📈 Epoch {epoch+1}/{cfg.training.epochs}")

        # ------------------------
        # Training Phase
        # ------------------------
        model.train()
        train_loss = 0.0

        for i, (images, targets) in enumerate(train_dataloader):
            if images is None or targets is None:
                continue

            model_need = "Tensor"

            images, processed_targets = model_input_format(images, targets, model_need, device, debug_mode=debug_mode)

            logger.debug(f"Type Images: {type(images)}")
            logger.debug(f"Images: {images}")
            logger.debug(f"Type Processed Targets: {type(processed_targets)}")
            logger.debug(f"Processed Targets: {processed_targets}")

            ########    WIE ARCHITEKTUR??????
            optimizer.zero_grad()
            model_output = model(images, processed_targets)
            loss_dict = model_output  # model_output to loss_dict converter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                # logger.info(f"   Batch {i+1}/{len(train_dataloader)}, Loss: {losses.item():.4f}, GradNorm: {gnorm if 'gnorm' in locals() else 'n/a'}")
                pass
            # --- Train Image-Logging (erste Batch, alle n Epochen) ---
            if i == 0 and (epoch % log_images_every_n_epochs == 0):
                with torch.no_grad():
                    model.eval()
                    if model_need == "Tensor":
                        imgs_vis = [img.detach().cpu() for img in images[:4]]
                        preds = model(images[:4])
                    else:
                        imgs_vis = [img.detach().cpu() for img in images[:4]]
                        preds = model([img.to(device) for img in images[:4]])
                    model.train()
                grid = make_gt_vs_pred_grid(imgs_vis, processed_targets[: len(imgs_vis)], preds, debug_mode=debug_mode)
                writer.add_image("Train/GT_vs_Pred", grid, epoch)

        avg_train_loss = train_loss / len(train_dataloader)

        # Parameter-Histogramme (einmal pro Epoche)
        for name, p in model.named_parameters():
            writer.add_histogram(f"Params/{name}", p.detach().cpu().numpy(), epoch)

        # =============================================================================
        # VAL EVALUATION (Loss)
        # =============================================================================
        logger.info("🧪 Starting val evaluation (loss)")
        model.train()  # (bewusst: Loss-Forward in train(), aber mit no_grad)
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                if images is None or targets is None:
                    continue
                images, processed_targets = model_input_format(images, targets, model_need, device, debug_mode=debug_mode)
                loss_dict = model(images, processed_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(1, len(val_dataloader))
        logger.info(f"🧪 Val Loss: {avg_val_loss:.4f}")
        # Hinweis: Epochen-Index für TB - hier 'epoch' statt 'cfg.training.epochs' sinnvoller,
        # sonst wird immer in dieselbe Step-Nummer geschrieben.
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)

        # =============================================================================
        # COCO EVALUATION (Metrics)
        # =============================================================================
        def _get(m: dict, key: str, default=None):
            return m[key] if (m is not None and key in m) else default

        def _get_float(m: dict, key: str):
            v = _get(m, key, None)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        m = evaluate_coco_from_loader(
            model=model,
            data_loader=val_dataloader,
            device=device,
            iou_type="bbox",
            input_format=model_need,
        )

        if m is not None:
            # --- Gesamtskalare, nur loggen wenn vorhanden ---
            for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"):
                v = _get_float(m, k)
                if v is not None and not (isinstance(v, float) and (v != v)):  # kein NaN
                    writer.add_scalar(f"COCO/{k}", v, epoch)

            # --- Per-Class (robust) ---
            per_class = _get(m, "per_class", {}) or {}
            pc_AP = per_class.get("AP", [])
            pc_AR = per_class.get("AR", [])
            cat_ids = per_class.get("cat_ids", list(range(len(pc_AP))))  # Fallback

            # Nur loggen, wenn überhaupt Werte vorhanden sind
            if isinstance(pc_AP, (list, tuple)) and len(pc_AP) > 0:
                pc_AP = np.array(pc_AP, dtype=float)
                writer.add_histogram("COCO/PerClass/AP", pc_AP, epoch)
            if isinstance(pc_AR, (list, tuple)) and len(pc_AR) > 0:
                pc_AR = np.array(pc_AR, dtype=float)
                writer.add_histogram("COCO/PerClass/AR", pc_AR, epoch)

            # Einzelwerte je Klasse (nur, wenn Dimensionen passen)
            if isinstance(cat_ids, (list, tuple)) and len(cat_ids) == len(pc_AP) == len(pc_AR):
                for cat_id, ap, ar in zip(cat_ids, pc_AP, pc_AR):
                    if ap == ap:  # not NaN
                        writer.add_scalar(f"COCO/PerClass/AP/{cat_id}", float(ap), epoch)
                    if ar == ar:
                        writer.add_scalar(f"COCO/PerClass/AR/{cat_id}", float(ar), epoch)

            writer.flush()
        else:
            logger.warning("COCO-Evaluation lieferte kein Ergebnis (m is None) - überspringe TB-Logging.")

        # --- Scheduler Step ---
        if use_scheduler:
            if cfg.scheduler.type == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            logger.info(f"📈 Current Learning Rate: {current_lr:.6f}")

        # # --- Epoch-Level Scalars ---
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
            logger.info(f"⏰ Early stopping patience counter: {patience_counter}/{cfg.training.early_stopping_patience}")
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info("⏹️ Early stopping triggered.")
                break

    # =============================================================================
    # TEST EVALUATION (Loss)
    # =============================================================================
    logger.info("🧪 Starting test evaluation (loss)")
    model.load_state_dict(torch.load(f"{experiment_dir}/models/best_model_weights.pth", weights_only=True))
    model.train()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in val_dataloader:
            if images is None or targets is None:
                continue
            images, processed_targets = model_input_format(images, targets, model_need, device, debug_mode=debug_mode)

            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()
    avg_test_loss = test_loss / len(val_dataloader)
    logger.info(f"🧪 Test Loss: {avg_test_loss:.4f}")
    writer.add_scalar("Test/Loss", avg_test_loss, cfg.training.epochs)

    # =============================================================================
    # CONFUSION MATRIX AUF TESTDATEN (Detections + CM in TensorBoard)
    # =============================================================================
    logger.info("🧪 Building confusion matrix on test set")
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in val_dataloader:
            if images is None or targets is None:
                continue
            images, processed_targets = model_input_format(images, targets, model_need, device, debug_mode=debug_mode)

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

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_ext, interpolation="nearest", cmap="Blues", alpha=0.6)
    ax.set_title("Confusion Matrix (GT rows, Pred cols)\nLast column=FN, Last row=FP", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("Ground truth class", fontsize=12)

    # Text in jede Kachel schreiben
    for i in range(cm_ext.shape[0]):
        for j in range(cm_ext.shape[1]):
            text_color = "white" if cm_ext[i, j] > cm_ext.max() / 2 else "black"
            ax.text(j, i, str(int(cm_ext[i, j])), ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_classes + 1))
    ax.set_xticklabels([str(i) for i in range(num_classes)] + ["FP"])
    ax.set_yticklabels([str(i) for i in range(num_classes)] + ["FN"])
    plt.tight_layout()
    writer.add_figure("Test/ConfusionMatrix_IOU0.5", fig, global_step=0)
    plt.close(fig)

    # =============================================================================
    # EXPERIMENT SUMMARY
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
    # FINAL LOGGING
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


def draw_boxes_on_img(img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # img: CHW uint8, boxes: xyxy
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    box_colors = [colors[int(l) % len(colors)] for l in labels]
    if boxes.numel() == 0:
        return draw_bounding_boxes(img, torch.zeros((0, 4), dtype=torch.int64), labels=[], width=2, colors=box_colors)
    return draw_bounding_boxes(img, boxes.round().to(torch.int64), labels=[str(int(l)) for l in labels], width=2, colors=box_colors)


def tensor_to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to uint8 format for visualization with improved range handling."""
    x = img.detach().cpu()

    # Already in uint8 format
    if x.dtype == torch.uint8:
        return x

    # Check tensor range
    min_val, max_val = float(x.min().item()), float(x.max().item())

    # Handle black or nearly black images
    if max_val <= min_val + 1e-5:
        return torch.zeros_like(x, dtype=torch.uint8)

    # Scale to [0,255] based on actual content range rather than assumptions
    if max_val <= 1.5:  # Likely [0,1] range
        x = (x * 255.0).clamp(0, 255)
    else:  # Likely already in high range
        x = x.clamp(0, 255)

    return x.to(torch.uint8)


def make_gt_vs_pred_grid(imgs_vis: List[torch.Tensor], targets_list, preds_list, debug_mode=False):
    """Create a grid of ground truth vs prediction images with better error handling."""
    panels = []

    for b_idx in range(len(imgs_vis)):
        img = imgs_vis[b_idx]

        # Skip invalid images
        if img is None or img.numel() == 0 or torch.isnan(img).any():
            logger.warning(f"Skipping invalid image at index {b_idx}")
            continue

        # Convert to uint8 for visualization
        img_u8 = tensor_to_uint8(img)

        if debug_mode:
            _check_img_range(img=img_u8, img_id=b_idx)

        # Process GT
        gt_t = targets_list[b_idx]
        gt_boxes_xyxy = gt_t["boxes"].detach().cpu()
        gt_labels = gt_t["labels"].detach().cpu()
        gt_img = draw_boxes_on_img(img_u8, gt_boxes_xyxy, gt_labels)

        if debug_mode:
            _check_img_range(img=gt_img, img_id=b_idx)

        # Process predictions
        p = preds_list[b_idx]
        p_boxes = p.get("boxes", torch.empty((0, 4))).detach().cpu()
        p_labels = p.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
        pred_img = draw_boxes_on_img(img_u8, p_boxes, p_labels)

        if debug_mode:
            _check_img_range(img=pred_img, img_id=b_idx)

        # Create side-by-side panel
        panel = torch.cat([gt_img, pred_img], dim=2)
        panels.append(panel)

    # Handle case where no valid panels could be created
    if not panels:
        logger.warning("No valid panels could be created for visualization")
        return torch.zeros((3, 64, 128), dtype=torch.uint8)  # Return small blank image

    # Create the final grid
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


def _get_img_wh_from_target(tgt: Dict[str, Any], img_tensor: Optional[torch.Tensor]) -> Tuple[int, int]:
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
) -> COCO:
    """
    Erstellt ein COCO-GT-Objekt (in-memory) aus dem DataLoader.
    Erwartet Targets mit:
      - "image_id" (Tensor oder int),
      - "boxes" (XYWH; evtl. normalisiert),
      - "labels" (contiguous ids, die über contig_to_catid auf COCO cat_id gemappt werden),
      - optional: "width"/"height" oder "orig_size"/"size".
    """
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1  # laufende ID

    # 1) Labels sammeln, um Kategorien 1..K zu definieren
    seen_labels = set()
    for batch in data_loader:
        imgs, tgts = batch
        for tgt in tgts:
            lbs = tgt["labels"]
            if isinstance(lbs, torch.Tensor):
                lbs = lbs.cpu().tolist()
            else:
                lbs = list(lbs)
            seen_labels.update(int(l) for l in lbs)

    categories = [{"id": int(l), "name": str(int(l))} for l in sorted(seen_labels)]

    # 2) GT aufbauen
    for batch in data_loader:
        imgs, tgts = batch

        for img, tgt in zip(imgs, tgts):
            img_id = int(tgt["image_id"].item() if isinstance(tgt["image_id"], torch.Tensor) else tgt["image_id"])
            img_w, img_h = _get_img_wh_from_target(tgt, img)

            images.append(
                {
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )

            gt_boxes = tgt["boxes"]
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.clone().cpu()
            else:
                gt_boxes = torch.as_tensor(gt_boxes)

            # Falls normalisiert (selten), in Pixel skalieren
            if gt_boxes.numel() > 0 and float(gt_boxes.max()) <= 1.5 and img_w > 0 and img_h > 0:
                scale_xyxy = torch.tensor([img_w, img_h, img_w, img_h], dtype=gt_boxes.dtype, device=gt_boxes.device)
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
                coco_cat = int(l)  # bereits 1..K
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "bbox": [x, y, w, h],
                        "area": float(max(w, 0.0) * max(h, 0.0)),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    coco = COCO()
    coco.dataset = {
        "info": {"description": "in-memory GT (1..K)", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco.createIndex()
    return coco


@torch.no_grad()
def evaluate_coco_from_loader(model, data_loader, device, iou_type="bbox", input_format="List"):
    model.eval()

    results = []
    coco_images, coco_annotations = [], []
    categories = set()
    ann_id = 1

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if input_format == "Tensor":
            images_device = torch.stack([img.to(device) for img in images])
            logger.debug(f"Input format is Tensor")
            logger.debug(f"Type images_device: {type(images_device)}")
            logger.debug(f"images_device: {images_device}")
            outputs = model(images_device)
        else:
            images_device = [img.to(device) for img in images]
            logger.debug(f"Input format is List")
            logger.debug(f"Type images_device: {type(images_device)}")
            logger.debug(f"images_device: {images_device}")
            outputs = model(images_device)

        # Debug-Ausgaben zum Output-Typ
        logger.debug(f"eval outputs type={type(outputs)}; len={len(outputs) if hasattr(outputs,'__len__') else 'n/a'}")
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], dict):
            logger.debug(f"eval outputs[0] keys={list(outputs[0].keys())}")
            logger.debug(f"outputs[0]: {outputs[0]}")
            pass
        else:
            if isinstance(outputs, dict):
                logger.warning("Model eval returned dict (vermutlich Loss-Dict). Erwartet: list[dict] mit 'boxes'. Es werden keine Detections erzeugt.")
            elif torch.is_tensor(outputs):
                logger.warning("Model eval returned Tensor. Erwartet: list[dict] pro image. Es werden keine Detections erzeugt.")
            else:
                logger.warning(f"Unerwarteter eval-Output: {type(outputs)}")

        # Wenn Output nicht der erwartete Typ ist, nächste Batch
        if not isinstance(outputs, (list, tuple)) or (len(outputs) != len(images)):
            logger.warning(f"Outputs nicht mit Batch kompatibel (len(outputs)={len(outputs) if hasattr(outputs,'__len__') else 'n/a'} vs len(images)={len(images)}). Batch wird übersprungen.")
            continue
        for i, (target, output) in enumerate(zip(targets, outputs)):
            # image-Metadaten
            _, H, W = images[i].shape
            image_id = int(target["image_id"].item() if torch.is_tensor(target["image_id"]) else target["image_id"])

            coco_images.append({"id": image_id, "width": int(W), "height": int(H)})

            # ---------- Ground Truth (XYWH bereits gegeben) ----------
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)  # erwartet XYWH
            gt_xywh = gt_boxes.cpu().numpy()
            gt_labels = torch.as_tensor(target["labels"]).cpu().numpy().astype(int)

            # optionale Felder robust auslesen
            if "area" in target and target["area"] is not None:
                gt_area = torch.as_tensor(target["area"], dtype=torch.float32).cpu().numpy()
            else:
                # Falls keine Area gegeben: aus w*h berechnen
                gt_area = (gt_boxes[:, 2] * gt_boxes[:, 3]).cpu().numpy()

            if "iscrowd" in target and target["iscrowd"] is not None:
                gt_iscrowd = torch.as_tensor(target["iscrowd"]).cpu().numpy().astype(int)
            else:
                gt_iscrowd = np.zeros((gt_xywh.shape[0],), dtype=np.int64)

            for lbl in gt_labels.tolist():
                categories.add(int(lbl))

            for box, label, area, crowd in zip(gt_xywh, gt_labels, gt_area, gt_iscrowd):
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],  # XYWH
                        "area": float(area),
                        "iscrowd": int(crowd),
                    }
                )
                ann_id += 1

            # ---------- Predictions (typisch XYXY -> XYWH) ----------
            if output is None or ("boxes" not in output):
                logger.warning(f"Model output for image {image_id} is None or missing 'boxes'. Skipping this image.")
                continue
            pred_xyxy = output["boxes"].detach().cpu()
            pred_scores = output.get("scores", torch.ones(len(pred_xyxy))).detach().cpu().numpy().astype(float)
            pred_labels = output.get("labels", torch.zeros(len(pred_xyxy), dtype=torch.long)).detach().cpu().numpy().astype(int)

            if pred_xyxy.numel() > 0:
                x1, y1, x2, y2 = pred_xyxy.unbind(-1)
                w = (x2 - x1).clamp(min=0)
                h = (y2 - y1).clamp(min=0)
                pred_xywh = torch.stack([x1, y1, w, h], dim=-1).cpu().numpy().astype(float)
            else:
                pred_xywh = np.zeros((0, 4), dtype=np.float32)

            for lbl in pred_labels.tolist():
                categories.add(int(lbl))

            for box, score, label in zip(pred_xywh, pred_scores, pred_labels):
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],  # XYWH
                        "score": float(score),
                    }
                )

    # ---------- COCO GT-Objekt ----------
    coco_gt_dict = {
        "info": {"description": "Autogenerated GT", "version": "1.0", "year": 2025, "contributor": "Wes", "date_created": "2025-08-19"},
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": cid, "name": str(cid)} for cid in sorted(categories)] or [{"id": 1, "name": "1"}],
    }
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    # ---------- COCOeval ----------
    logger.debug(f"Type Model output: {type(outputs)}")
    logger.debug(f"Model output: {outputs}\n\n")
    logger.debug(f"Type Results: {type(results)}")
    logger.debug(f"Results: {results}\n\n")
    logger.debug(f"Type COCO GT: {type(coco_gt)}")
    logger.debug(f"COCO GT: {coco_gt}\n\n")

    if len(results) == 0:
        logger.warning("COCO eval: Keine Detections erzeugt - gebe Null-Metriken zurück und überspringe COCOeval.")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

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


def model_input_format(images, targets, model_need, device, debug_mode=False):
    def to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        # Eingabe immer xywh -> Ziel xyxy
        return coco_xywh_to_xyxy(boxes)

    if model_need == "Tensor":
        images = torch.stack([image.to(device) for image in images])

        processed_targets = []
        for tgt in targets:
            if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                boxes = to_xyxy(boxes)
                processed_targets.append({"boxes": boxes.to(device).float(), "labels": labels.to(device).long()})

            elif isinstance(tgt, list) and len(tgt) > 0:
                boxes_xywh = torch.tensor([ann["bbox"] for ann in tgt], dtype=torch.float32)
                labels = torch.tensor([ann["category_id"] for ann in tgt], dtype=torch.long)
                boxes = to_xyxy(boxes_xywh)
                processed_targets.append({"boxes": boxes.to(device), "labels": labels.to(device)})

            else:
                processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})

    elif model_need == "List":
        images = [img.to(device) for img in images]

        processed_targets = []
        for tgt in targets:
            if isinstance(tgt, dict) and "boxes" in tgt and "labels" in tgt:
                boxes = tgt["boxes"] if isinstance(tgt["boxes"], torch.Tensor) else torch.as_tensor(tgt["boxes"], dtype=torch.float32)
                labels = tgt["labels"] if isinstance(tgt["labels"], torch.Tensor) else torch.as_tensor(tgt["labels"], dtype=torch.long)
                boxes = to_xyxy(boxes)
                processed_targets.append({"boxes": boxes.to(device).float(), "labels": labels.to(device).long()})

            elif isinstance(tgt, list) and len(tgt) > 0:
                boxes_xywh = torch.tensor([ann["bbox"] for ann in tgt], dtype=torch.float32)
                labels = torch.tensor([ann["category_id"] for ann in tgt], dtype=torch.long)
                boxes = to_xyxy(boxes_xywh)
                processed_targets.append({"boxes": boxes.to(device), "labels": labels.to(device)})

            else:
                processed_targets.append({"boxes": torch.empty((0, 4), device=device), "labels": torch.empty((0,), dtype=torch.long, device=device)})

    return images, processed_targets


def debug_show(img, title="Debug Image", enable=False):
    if not enable:
        return
    plt.figure()
    if isinstance(img, torch.Tensor):
        if img.dim() == 3 and img.shape[0] in [1, 3]:
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        else:
            plt.imshow(img)
    else:
        plt.imshow(np.array(img))
    plt.title(title)
    plt.axis("off")
    plt.show()


def debug_show_grid(images, titles=None, rows=None, cols=None, figsize=(15, 10), enable=False):
    """
    Display multiple images in a grid layout in a single window.

    Args:
        images: List of images (PIL Images or torch Tensors)
        titles: Optional list of titles for each image
        rows: Number of rows in grid (calculated automatically if None)
        cols: Number of columns in grid (calculated automatically if None)
        figsize: Figure size (width, height) in inches
        enable: Whether to actually display the images (for easy disabling)
    """
    if not enable:
        return

    n_images = len(images)
    if n_images == 0:
        return

    if rows is None and cols is None:
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
    elif rows is None:
        rows = (n_images + cols - 1) // cols
    elif cols is None:
        cols = (n_images + rows - 1) // rows

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for i, img in enumerate(images):
        if i >= len(axs):
            break

        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                # Convert CHW to HWC and ensure it's on CPU
                img_np = img.permute(1, 2, 0).cpu().numpy()

                # Handle normalization - rescale if needed
                if img_np.max() <= 1.0:
                    img_np = img_np

                axs[i].imshow(img_np)
            else:
                axs[i].imshow(img.cpu().numpy())
        else:
            axs[i].imshow(np.array(img))

        axs[i].set_axis_off()
        if titles and i < len(titles):
            axs[i].set_title(titles[i])

    # Hide empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show(block=False)  # Use block=False to allow code to continue
    plt.pause(0.1)  # Small pause to ensure rendering

    return fig


def _check_img_range(img, img_id="unknown"):
    """
    Überprüft den range eines Images, erkennt schwarze Images und gibt entsprechende Logmeldungen aus.

    Args:
        img: Das zu überprüfende Image (torch.Tensor, PIL.Image oder numpy.ndarray)
        img_id: Identifikator für das Image (für Logging)
    """
    try:
        # Tensor-Verarbeitung
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                min_val = img.min().item()
                max_val = img.max().item()
            else:
                logger.warning(f"Unerwartete Tensor-Form: {img.shape}")
                return

        # PIL Image-Verarbeitung
        elif hasattr(img, "getextrema"):  # PIL Image
            extrema = img.getextrema()
            if isinstance(extrema[0], tuple):  # RGB oder anderer Mehrkanal
                min_val = min(ext[0] for ext in extrema)
                max_val = max(ext[1] for ext in extrema)
            else:  # Graustufen
                min_val = extrema[0]
                max_val = extrema[1]

        # NumPy oder andere Imagetypen
        else:
            # Fallback zu NumPy
            if isinstance(img, np.ndarray):
                np_img = img
            else:
                np_img = np.array(img)

            min_val = np_img.min()
            max_val = np_img.max()

        # Bestimme den Imagetype basierend auf dem Range
        if max_val < 1e-5:
            logger.warning(f"Image {img_id}: SCHWARZ (min={min_val:.6f}, max={max_val:.6f})")
        elif max_val <= 1.5:  # Toleranz für kleine Rundungsfehler
            logger.debug(f"Image {img_id}: Range [0,1] (min={min_val:.6f}, max={max_val:.6f})")
        elif max_val <= 255.5:  # Toleranz für kleine Rundungsfehler
            logger.debug(f"Image {img_id}: Range [0,255] (min={min_val:.6f}, max={max_val:.6f})")
        else:
            logger.warning(f"Image {img_id}: UNGÜLTIGER Range (min={min_val:.6f}, max={max_val:.6f})")

    except Exception as e:
        logger.error(f"Fehler bei der Überprüfung des Imageranges für Image {img_id}: {e}")


if __name__ == "__main__":
    main()
