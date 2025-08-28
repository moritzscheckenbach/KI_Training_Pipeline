# pip install streamlit ruamel.yaml pandas
# python -m pip install fsspec

import io
import os
import sys
import time
import re
import subprocess
from pathlib import Path

import streamlit as st
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

# =========================
# Configuration Variables
# =========================
DATASET_ALREADY_SPLIT = False  # Set to True if your datasets are already split into train/valid/test folders

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# =========================
# Helpers
# =========================
def quote_specific_strings(data):
    """Only set specific strings in quotation marks and specific arrays as flow-style"""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in ["file", "type", "path", "trans_file", "strategy", "root", "head_name", "backbone_name", "mode"]:
                if isinstance(v, str) and v:
                    result[k] = DoubleQuotedScalarString(v)
                else:
                    result[k] = v
            elif k in ["betas", "freeze_layers", "unfreeze_layers", "milestones"]:
                if isinstance(v, list):
                    flow_seq = CommentedSeq(v)
                    flow_seq.fa.set_flow_style()
                    if k in ["freeze_layers", "unfreeze_layers"] and all(isinstance(item, str) for item in v):
                        flow_seq[:] = [DoubleQuotedScalarString(item) if item else item for item in v]
                    result[k] = flow_seq
                else:
                    result[k] = v
            else:
                result[k] = quote_specific_strings(v)
        return result
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return [DoubleQuotedScalarString(item) if item else item for item in data]
        else:
            return [quote_specific_strings(item) for item in data]
    else:
        return data


def dump_yaml_str(data: dict) -> str:
    buf = io.StringIO()
    quoted_data = quote_specific_strings(data)
    yaml.default_flow_style = False
    yaml.dump(quoted_data, buf)
    return buf.getvalue()


def list_dirs(path: Path) -> list[str]:
    try:
        return [p.name for p in sorted(path.iterdir()) if p.is_dir()]
    except Exception:
        return []


def list_files(path: Path, suffix=".py") -> list[str]:
    try:
        return [p.stem for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() == suffix and p.name != "__init__.py"]
    except Exception:
        return []


def get_dataset_options(datasets_root: Path, task: str) -> list[str]:
    task_path = datasets_root / task
    if not task_path.exists():
        return []
    options = []
    try:
        for type_folder in sorted(task_path.iterdir()):
            if type_folder.is_dir():
                for dataset in sorted(type_folder.iterdir()):
                    if dataset.is_dir():
                        options.append(f"{type_folder.name}/{dataset.name}")
    except Exception:
        pass
    return options


def get_num_classes(datasets_root: Path, task: str, dataset_path: str) -> int:
    try:
        classes_file = datasets_root / task / dataset_path / "classes.yaml"
        if classes_file.exists():
            with classes_file.open("r", encoding="utf-8") as f:
                classes_data = yaml.load(f)
                return classes_data.get("num_classes", 1)
    except Exception:
        pass
    return 1


def check_dataset_split_status(datasets_root: Path, task: str, dataset_path: str) -> bool:
    if not dataset_path:
        return False
    dataset_full_path = datasets_root / task / dataset_path
    if not dataset_full_path.exists():
        return False
    if (dataset_full_path / "train").exists():
        return True
    if (dataset_full_path / "dataset").exists():
        return False
    return False


# ---------- Runtime/Logging helpers ----------
def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _ensure_dirs():
    Path("conf").mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)


def _tail_file(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                _ = f.readline()
            return f.read()
    except Exception:
        return ""


# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Training Pipeline Config", layout="wide")
st.title("Training Pipeline Configuration")

# =========================
# Base folders
# =========================
datasets_root = Path("datasets")
augmentations_root = Path("augmentations")

# =========================
# Task Selection
# =========================
st.subheader("1. Select Task")
task_options = ["classification", "object_detection", "segmentation"]
task = st.selectbox("Task", task_options, index=0)

# =========================
# Dataset Selection
# =========================
st.subheader("2. Select Dataset")
dataset_options = get_dataset_options(datasets_root, task)
if not dataset_options:
    st.info(f"No datasets found in 'datasets/{task}/'.")
    dataset = None
    DATASET_ALREADY_SPLIT = False
else:
    dataset = st.selectbox("Dataset", dataset_options)
    DATASET_ALREADY_SPLIT = check_dataset_split_status(datasets_root, task, dataset)

# Dataset Split Configuration
st.markdown("**Dataset Split Configuration**")
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

if not DATASET_ALREADY_SPLIT:
    st.info("📊 Dataset will be automatically split during training")
    st.markdown("**Define Split Ratios**")
    train_ratio = st.slider("Train Ratio", min_value=0.1, max_value=0.9, value=0.70, step=0.01)
    remaining = 1.0 - train_ratio
    val_ratio = st.slider("Val Ratio", min_value=0.0, max_value=remaining, value=min(0.15, remaining), step=0.01)
    test_ratio = remaining - val_ratio

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Ratio", f"{train_ratio:.3f}")
    with col2:
        st.metric("Val Ratio", f"{val_ratio:.3f}")
    with col3:
        st.metric("Test Ratio", f"{test_ratio:.3f}")
else:
    st.info("📁 Using pre-split dataset folders (train/valid/test)")

# =========================
# Augmentation Selection
# =========================
st.subheader("3. Select Augmentation")
augmentation_options = list_files(augmentations_root, ".py")
if not augmentation_options:
    st.info("No augmentation files found in 'augmentations/'.")
    augmentation = None
else:
    augmentation = st.selectbox("Augmentation", augmentation_options)

# =========================
# Mode Selection
# =========================
st.subheader("4. Training Mode")
mode_options = ["Training from Architecture", "Transfer Learning: Self-trained Model"]
mode = st.radio("Select Mode", mode_options, horizontal=True)
debug_option = st.selectbox("Debug Mode", ["No", "Yes"], index=0)
modus_debug = True if debug_option == "Yes" else False

# =========================
# Architecture/Model Selection based on Mode
# =========================
architecture = None
model = None
head_name = "detection_head"
backbone_name = "backbone"

if mode == "Training from Architecture":
    st.subheader("5. Select Architecture")
    architecture_root = Path("model_architecture") / task
    architecture_options = list_files(architecture_root, ".py")
    if not architecture_options:
        st.info(f"No architecture files found in 'model_architecture/{task}/'.")
        architecture = None
    else:
        architecture = st.selectbox("Architecture", architecture_options)

elif mode == "Transfer Learning: Self-trained Model":
    st.subheader("5. Select Model")
    model_root = Path("trained_models") / task
    model_options = list_dirs(model_root)
    if not model_options:
        st.info(f"No models found in 'trained_models/{task}/'.")
        model = None
    else:
        model = st.selectbox("Model", model_options)
        head_name = st.text_input("Head Name", value="detection_head")
        backbone_name = st.text_input("Backbone Name", value="backbone")

# =========================
# Training Configuration
# =========================
st.subheader("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Parameters**")
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=5)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=16)
    learning_rate = st.number_input("Learning Rate", min_value=1e-8, max_value=1.0, value=0.001, format="%.6f")
    early_stopping_patience = st.number_input("Early Stopping Patience", min_value=1, max_value=100, value=10)
    random_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42)

with col2:
    st.markdown("**Scheduler Parameters**")
    scheduler_type = st.selectbox("Scheduler Type", ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    scheduler_params = {}
    if scheduler_type == "StepLR":
        scheduler_params["step_size"] = st.number_input("Step Size (epochs)", 1, 500, 10)
        scheduler_params["gamma"] = st.number_input("Gamma", 0.01, 1.0, 0.1, format="%.3f")
    elif scheduler_type == "MultiStepLR":
        milestones_str = st.text_input("Milestones (comma-separated)", "30,60,90")
        scheduler_params["milestones"] = [int(x.strip()) for x in milestones_str.split(",") if x.strip().isdigit()]
        scheduler_params["gamma"] = st.number_input("Gamma", 0.01, 1.0, 0.1, format="%.3f")
    elif scheduler_type == "ExponentialLR":
        scheduler_params["gamma"] = st.number_input("Gamma", 0.01, 1.0, 0.95, format="%.3f")
    elif scheduler_type == "CosineAnnealingLR":
        scheduler_params["T_max"] = st.number_input("T_max (epochs)", 1, 1000, 50)
        scheduler_params["eta_min"] = st.number_input("Eta Min (min LR)", 1e-10, 1e-3, 1e-6, format="%.2e")
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler_params["mode"] = st.selectbox("Mode", ["min", "max"])
        scheduler_params["factor"] = st.number_input("Factor (LR *= factor)", 0.01, 1.0, 0.5, format="%.3f")
        scheduler_params["patience"] = st.number_input("Patience (epochs)", 1, 100, 5)
        scheduler_params["threshold"] = st.number_input("Threshold", 0.0, 1.0, 1e-4, format="%.1e")
        scheduler_params["cooldown"] = st.number_input("Cooldown (epochs)", 0, 100, 0)
        scheduler_params["min_lr"] = st.number_input("Min Learning Rate", 1e-10, 1e-3, 1e-6, format="%.2e")

# =========================
# Optimizer Configuration
# =========================
st.markdown("**Optimizer Parameters**")

col1, col2, col3 = st.columns(3)

with col1:
    optimizer_type = st.selectbox("Optimizer Type", ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta"])
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=0.01, format="%.4f")

with col2:
    if optimizer_type in ["Adam", "AdamW"]:
        st.markdown("**Adam Parameters**")
        beta1 = st.number_input("Beta 1", min_value=0.0, max_value=1.0, value=0.9, format="%.3f")
        beta2 = st.number_input("Beta 2", min_value=0.0, max_value=1.0, value=0.999, format="%.3f")

with col3:
    if optimizer_type == "SGD":
        st.markdown("**SGD Parameters**")
        momentum = st.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.9, format="%.3f")

# =========================
# Model Configuration (only for Transfer Learning)
# =========================
if mode == "Transfer Learning: Self-trained Model":
    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Freezing Strategy**")
        freezing_enabled = st.selectbox("Freezing Enabled", [True, False])

        if freezing_enabled:
            freezing_strategy = st.selectbox("Freezing Strategy", ["freeze_all_except_head", "freeze_early_layers", "freeze_backbone", "unfreeze_all", "custom_freeze"])

            if freezing_strategy == "freeze_early_layers":
                freeze_until_layer = st.number_input("Freeze Until Layer. CAUTION! Layers must be children of Backbone.", min_value=1, max_value=50, value=6)

            if freezing_strategy == "custom_freeze":
                st.markdown("**Custom Freeze Settings**")
                freeze_layers_input = st.text_input("Freeze Layers (comma separated)", value="backbone.layer1, backbone.layer2")
                unfreeze_layers_input = st.text_input("Unfreeze Layers (comma separated)", value="detection_head, backbone.layer4")
                freeze_layers = [layer.strip() for layer in freeze_layers_input.split(",") if layer.strip()]
                unfreeze_layers = [layer.strip() for layer in unfreeze_layers_input.split(",") if layer.strip()]
            else:
                freeze_layers = []
                unfreeze_layers = []

    with col2:
        st.markdown("**Learning Rate Multipliers**")
        backbone_lr_multiplier = st.number_input("Backbone LR Multiplier", min_value=0.001, max_value=10.0, value=0.1, format="%.3f")
        head_lr_multiplier = st.number_input("Head LR Multiplier", min_value=0.001, max_value=10.0, value=1.0, format="%.3f")

# =========================
# Config Generation
# =========================
def _get(d, key, default, cast=lambda x: x):
    return cast(d.get(key, default))

DEFAULTS = {
    "step_size": 10,
    "gamma": 0.1,
    "milestones": [30, 60, 90],
    "exp_gamma": 0.95,
    "T_max": 50,
    "eta_min": 1e-6,
    "mode": "min",
    "factor": 0.5,
    "patience": 5,
    "threshold": 1e-4,
    "cooldown": 0,
    "min_lr": 1e-6,
}

dataset_type = ""
if dataset:
    dataset_type = dataset.split("/")[0]

num_classes = get_num_classes(datasets_root, task, dataset) if dataset else 1

config = {
    "training": {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "early_stopping_patience": int(early_stopping_patience),
        "random_seed": int(random_seed),
        "debug_mode": bool(modus_debug),
    },
    "scheduler": {
        "file": "scheduler",
        "type": scheduler_type,
        "step_size": _get(scheduler_params, "step_size", DEFAULTS["step_size"], int),
        "gamma": float(scheduler_params["gamma"]) if "gamma" in scheduler_params else (DEFAULTS["exp_gamma"] if scheduler_type == "ExponentialLR" else DEFAULTS["gamma"]),
        "milestones": _get(scheduler_params, "milestones", DEFAULTS["milestones"], list),
        "T_max": _get(scheduler_params, "T_max", DEFAULTS["T_max"], int),
        "eta_min": _get(scheduler_params, "eta_min", DEFAULTS["eta_min"], float),
        "mode": _get(scheduler_params, "mode", DEFAULTS["mode"], str),
        "factor": _get(scheduler_params, "factor", DEFAULTS["factor"], float),
        "patience": _get(scheduler_params, "patience", DEFAULTS["patience"], int),
        "threshold": _get(scheduler_params, "threshold", DEFAULTS["threshold"], float),
        "cooldown": _get(scheduler_params, "cooldown", DEFAULTS["cooldown"], int),
        "min_lr": _get(scheduler_params, "min_lr", DEFAULTS["min_lr"], float),
    },
    "optimizer": {
        "type": optimizer_type,
        "weight_decay": float(weight_decay),
        "adam": {"betas": [float(beta1) if optimizer_type in ["Adam", "AdamW"] else 0.9, float(beta2) if optimizer_type in ["Adam", "AdamW"] else 0.999]},
        "sgd": {"momentum": float(momentum) if optimizer_type == "SGD" else 0.9},
    },
    "augmentation": {"file": augmentation if augmentation else ""},
    "model": {
        "type": task,
        "file": architecture if architecture else "",
        "transfer_learning": {
            "enabled": mode == "Transfer Learning: Self-trained Model",
            "path": f"trained_models/{task}/{model}/models/best_model_weights.pth" if model else "",
            "trans_file": "transferlearning",
            "head_name": head_name,
            "backbone_name": backbone_name,
            "freezing": {
                "enabled": bool('freezing_enabled' in locals() and freezing_enabled) if mode == "Transfer Learning: Self-trained Model" else False,
                "strategy": (freezing_strategy if 'freezing_strategy' in locals() and freezing_enabled else "freeze_all_except_head") if mode == "Transfer Learning: Self-trained Model" else "freeze_all_except_head",
                "freeze_early_layers": {
                    "freeze_until_layer": int(freeze_until_layer) if 'freeze_until_layer' in locals() and freezing_enabled else 6
                },
                "custom": {
                    "freeze_layers": (freeze_layers if 'freeze_layers' in locals() and freezing_enabled else ["backbone.layer1", "backbone.layer2"]),
                    "unfreeze_layers": (unfreeze_layers if 'unfreeze_layers' in locals() and freezing_enabled else ["detection_head", "backbone.layer4"]),
                },
            },
            "lr": {
                "backbone_lr_multiplier": float(backbone_lr_multiplier) if mode == "Transfer Learning: Self-trained Model" else 0.1,
                "head_lr_multiplier": float(head_lr_multiplier) if mode == "Transfer Learning: Self-trained Model" else 1.0,
            },
        },
    },
    "dataset": {
        "root": f"datasets/{task}/{dataset}/" if dataset else "",
        "type": dataset_type,
        "num_classes": num_classes,
        "autosplit": {"enabled": not DATASET_ALREADY_SPLIT, "train_ratio": float(train_ratio), "val_ratio": float(val_ratio), "test_ratio": float(test_ratio)},
    },
}

# =========================
# YAML Preview & Save
# =========================
st.markdown("---")
st.subheader("YAML Preview")
yaml_text = dump_yaml_str(config)
st.code(yaml_text, language="yaml")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Training"):
        _ensure_dirs()
        output_path = Path("conf/config.yaml")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(yaml_text)
        st.success(f"Saved: {output_path.resolve()}")

        training_files = {"classification": "training_class.py", "object_detection": "training_objdet.py", "segmentation": "training_seg.py"}
        training_file = training_files.get(task)

        if training_file and Path(training_file).exists():
            try:
                st.info(f"Starting training for {task} ...")

                # --- Subprozess + Logfile ---
                log_path = Path("runs/ui_training.log")
                # line-buffered schreiben; existierende Logs überschreiben
                log_f = log_path.open("w", encoding="utf-8", buffering=1)
                process = subprocess.Popen(
                    [sys.executable, training_file],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    env=os.environ.copy(),
                )
                st.session_state["train_pid"] = process.pid
                st.session_state["log_path"] = str(log_path)
                st.success(f"Training started (PID: {process.pid}). Logging to {log_path}")

            except Exception as e:
                st.error(f"Error starting training: {e}")
        else:
            st.error(f"Training file not found: {training_file}")

with col2:
    st.download_button("Download", data=yaml_text, file_name="config.yaml", mime="text/yaml")

# ========= Live Log Anzeige =========
if "log_path" in st.session_state:
    lp = Path(st.session_state["log_path"])
    with st.expander("Training logs (tail)"):
        st.caption(str(lp.resolve()))
        st.code(_tail_file(lp), language="bash")
        # Optional: Auto-Refresh alle 2s aktivieren (kommentiere IN oder AUS)
        # st.experimental_rerun() nicht verwenden, sonst Endlosschleife.
        if st.button("Refresh logs"):
            pass  # Streamlit rerun triggert automatisch und lädt neu
