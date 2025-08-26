# pip install streamlit ruamel.yaml pandas
# python -m pip install fsspec

import io
import re
from pathlib import Path

import streamlit as st
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

# =========================
# Helpers
# =========================
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def quote_specific_strings(data):
    """Nur bestimmte Strings in Anführungszeichen setzen und bestimmte Arrays als Flow-Style"""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Bestimmte Keys, deren Werte in Anführungszeichen sollen
            if k in ["file", "type", "path", "trans_file", "strategy", "root", "head_name", "backbone_name"]:
                if isinstance(v, str) and v:
                    result[k] = DoubleQuotedScalarString(v)
                else:
                    result[k] = v
            # Bestimmte Arrays sollen als Flow-Style (eckige Klammern) dargestellt werden
            elif k in ["betas", "freeze_layers", "unfreeze_layers"]:
                if isinstance(v, list):
                    flow_seq = CommentedSeq(v)
                    flow_seq.fa.set_flow_style()
                    # Strings in diesen Arrays auch quotieren
                    if k in ["freeze_layers", "unfreeze_layers"] and all(isinstance(item, str) for item in v):
                        flow_seq[:] = [DoubleQuotedScalarString(item) if item else item for item in v]
                    result[k] = flow_seq
                else:
                    result[k] = v
            else:
                result[k] = quote_specific_strings(v)
        return result
    elif isinstance(data, list):
        # Listen von Strings (außer den speziellen Flow-Style Listen) auch quotieren
        if all(isinstance(item, str) for item in data):
            return [DoubleQuotedScalarString(item) if item else item for item in data]
        else:
            return [quote_specific_strings(item) for item in data]
    else:
        return data


def dump_yaml_str(data: dict) -> str:
    buf = io.StringIO()
    quoted_data = quote_specific_strings(data)
    yaml.default_flow_style = False  # Ensure block style for lists
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
    """Durchsucht datasets/{task}/ nach Type-Ordnern und deren Datasets"""
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
    """Liest num_classes aus der classes.yaml Datei"""
    try:
        classes_file = datasets_root / task / dataset_path / "classes.yaml"
        if classes_file.exists():
            with classes_file.open("r", encoding="utf-8") as f:
                classes_data = yaml.load(f)
                return classes_data.get("num_classes", 1)
    except Exception:
        pass
    return 1


# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Training Pipeline Config", layout="wide")
st.title("Training Pipeline Configuration")

# =========================
# Basisordner
# =========================
datasets_root = Path("datasets")
augmentations_root = Path("augmentations")

# =========================
# Task Selection
# =========================
st.subheader("1. Task auswählen")
task_options = ["classification", "object_detection", "segmentation"]
task = st.selectbox("Task", task_options, index=0)

# =========================
# Dataset Selection
# =========================
st.subheader("2. Dataset auswählen")
dataset_options = get_dataset_options(datasets_root, task)
if not dataset_options:
    st.info(f"Keine Datasets in 'datasets/{task}/' gefunden.")
    dataset = None
else:
    dataset = st.selectbox("Dataset", dataset_options)

# Dataset Split Configuration
st.markdown("**Dataset Split Konfiguration**")
split_mode_options = ["Ja", "Nein"]
split_mode = st.radio("Datensatz bereits in train/valid/test gesplittet?", split_mode_options, horizontal=True)

# Default split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

if split_mode == "Nein":
    st.markdown("**Split Verhältnisse definieren**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_ratio = st.number_input("Train Ratio", min_value=0.0, max_value=1.0, value=0.70, format="%.2f")
    with col2:
        val_ratio = st.number_input("Val Ratio", min_value=0.0, max_value=1.0, value=0.15, format="%.2f")
    with col3:
        test_ratio = st.number_input("Test Ratio", min_value=0.0, max_value=1.0, value=0.15, format="%.2f")
    
    # Validation: Check if sum equals 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:  # Small tolerance for floating point errors
        st.error(f"⚠️ Die Summe der Verhältnisse muss 1.0 ergeben! Aktuell: {total_ratio:.3f}")
    else:
        st.success(f"✅ Split Verhältnisse korrekt: {total_ratio:.3f}")

# =========================
# Augmentation Selection
# =========================
st.subheader("3. Augmentation auswählen")
augmentation_options = list_files(augmentations_root, ".py")
if not augmentation_options:
    st.info("Keine Augmentation-Dateien in 'augmentations/' gefunden.")
    augmentation = None
else:
    augmentation = st.selectbox("Augmentation", augmentation_options)

# =========================
# Modus Selection
# =========================
st.subheader("4. Trainingsmodus")
mode_options = ["Training aus Architektur", "Transferlearning: selbstrainiertes Modell"]
mode = st.radio("Modus wählen", mode_options, horizontal=True)
debug_option = st.selectbox("Debug Modus", ["Nein", "Ja"], index=0)
modus_debug = True if debug_option == "Ja" else False

# =========================
# Architecture/Model Selection based on Mode
# =========================
architecture = None
model = None
# Default values
head_name = "detection_head"
backbone_name = "backbone"


if mode == "Training aus Architektur":
    st.subheader("5. Architektur auswählen")
    architecture_root = Path("model_architecture") / task
    architecture_options = list_files(architecture_root, ".py")
    if not architecture_options:
        st.info(f"Keine Architektur-Dateien in 'model_architecture/{task}/' gefunden.")
        architecture = None
    else:
        architecture = st.selectbox("Architektur", architecture_options)

elif mode == "Transferlearning: selbstrainiertes Modell":
    st.subheader("5. Modell auswählen")
    model_root = Path("trained_models") / task
    model_options = list_dirs(model_root)
    if not model_options:
        st.info(f"Keine Modelle in 'trained_models/{task}/' gefunden.")
        model = None
    else:
        model = st.selectbox("Modell", model_options)
        head_name = st.text_input("Name des Heads", value="detection_head")
        backbone_name = st.text_input("Name des Backbones", value="backbone")

# =========================
# Training Configuration
# =========================
st.subheader("Training Konfiguration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Parameter**")
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=5)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=16)
    learning_rate = st.number_input("Learning Rate", min_value=1e-8, max_value=1.0, value=0.001, format="%.6f")
    early_stopping_patience = st.number_input("Early Stopping Patience", min_value=1, max_value=100, value=10)
    random_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42)

with col2:
    st.markdown("**Scheduler Parameter**")
    scheduler_type = st.selectbox("Scheduler Type", ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    scheduler_patience = st.number_input("Scheduler Patience", min_value=1, max_value=100, value=5)
    scheduler_factor = st.number_input("Scheduler Factor", min_value=0.01, max_value=1.0, value=0.5, format="%.3f")
    min_lr = st.number_input("Min Learning Rate", min_value=1e-10, max_value=1e-3, value=1e-6, format="%.2e")

# =========================
# Optimizer Configuration
# =========================
st.markdown("**Optimizer Parameter**")

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
# Model Configuration (nur bei Transferlearning)
# =========================
if mode == "Transferlearning: selbstrainiertes Modell":
    st.subheader("Model Konfiguration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Freezing Strategy**")
        freezing_enabled = st.selectbox("Freezing Enabled", [True, False])

        if freezing_enabled:
            freezing_strategy = st.selectbox("Freezing Strategy", ["freeze_all_except_head", "freeze_early_layers", "freeze_backbone", "unfreeze_all", "custom_freeze"])

            if freezing_strategy == "freeze_early_layers":
                freeze_until_layer = st.number_input("Freeze Until Layer. ACHTUNG! Layers müssen children von Backbone sein.", min_value=1, max_value=50, value=6)

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

# Parse dataset type from dataset selection
dataset_type = ""
if dataset:
    dataset_type = dataset.split("/")[0]  # z.B. "Type_COCO" aus "Type_COCO/Test_Duckiebots"

# Get num_classes from classes.yaml
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
    "scheduler": {"file": "scheduler", "type": scheduler_type, "patience": int(scheduler_patience), "factor": float(scheduler_factor), "min_lr": float(min_lr)},
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
            "enabled": mode == "Transferlearning: selbstrainiertes Modell",
            "path": f"trained_models/{task}/{model}/models/best_model_weights.pth" if model else "",
            "trans_file": "transferlearning",
            "head_name": head_name,
            "backbone_name": backbone_name,
            "freezing": {
                "enabled": bool(freezing_enabled) if mode == "Transferlearning: selbstrainiertes Modell" else False,
                "strategy": freezing_strategy if mode == "Transferlearning: selbstrainiertes Modell" and freezing_enabled else "freeze_all_except_head",
                "freeze_early_layers": {
                    "freeze_until_layer": int(freeze_until_layer) if mode == "Transferlearning: selbstrainiertes Modell" and freezing_enabled and freezing_strategy == "freeze_early_layers" else 6
                },
                "custom": {
                    "freeze_layers": (
                        freeze_layers if mode == "Transferlearning: selbstrainiertes Modell" and freezing_enabled and freezing_strategy == "custom_freeze" else ["backbone.layer1", "backbone.layer2"]
                    ),
                    "unfreeze_layers": (
                        unfreeze_layers if mode == "Transferlearning: selbstrainiertes Modell" and freezing_enabled and freezing_strategy == "custom_freeze" else ["detection_head", "backbone.layer4"]
                    ),
                },
            },
            "lr": {
                "backbone_lr_multiplier": float(backbone_lr_multiplier) if mode == "Transferlearning: selbstrainiertes Modell" else 0.1,
                "head_lr_multiplier": float(head_lr_multiplier) if mode == "Transferlearning: selbstrainiertes Modell" else 1.0,
            },
        },
    },
    "dataset": {
        "root": f"datasets/{task}/{dataset}/" if dataset else "", 
        "type": dataset_type, 
        "num_classes": num_classes,
        "autosplit": {
            "enabled": not split_mode == "Ja",
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio)
        }
    },
}

# =========================
# YAML Preview & Save
# =========================
st.markdown("---")
st.subheader("YAML Vorschau")
yaml_text = dump_yaml_str(config)
st.code(yaml_text, language="yaml")

col1, col2 = st.columns(2)

with col1:
    if st.button("Training starten"):
        output_path = Path("conf/config.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(yaml_text)
        st.success(f"Gespeichert: {output_path.resolve()}")

        # Start the appropriate training file based on task
        training_files = {"classification": "training_class.py", "object_detection": "training_objdet.py", "segmentation": "training_seg.py"}

        training_file = training_files.get(task)
        if training_file and Path(training_file).exists():
            try:
                st.info(f"Starte Training für {task}...")
                # Start training in a new terminal window
                import shutil
                import subprocess
                import sys

                # Check which terminal is available and use the best option
                if shutil.which("xterm"):
                    # Basic xterm (always available) - ZUERST prüfen!
                    st.info("Verwende xterm...")
                    terminal_command = ["xterm", "-e", f"bash -c 'cd {Path.cwd()} && {sys.executable} {training_file}; echo \"Training beendet. Drücke Enter zum Schließen...\"; read'"]
                elif shutil.which("konsole"):
                    # KDE terminal
                    terminal_command = ["konsole", "-e", "bash", "-c", f"cd {Path.cwd()} && {sys.executable} {training_file}; echo 'Training beendet. Drücke Enter zum Schließen...'; read"]
                elif shutil.which("x-terminal-emulator"):
                    # Standard Linux terminal (might link to problematic gnome-terminal)
                    st.warning("Verwende x-terminal-emulator (könnte gnome-terminal sein)...")
                    terminal_command = ["x-terminal-emulator", "-e", "bash", "-c", f"cd {Path.cwd()} && {sys.executable} {training_file}; echo 'Training beendet. Drücke Enter zum Schließen...'; read"]
                elif shutil.which("gnome-terminal"):
                    # GNOME terminal (fallback)
                    st.warning("Verwende gnome-terminal (könnte Snap-Probleme haben)...")
                    terminal_command = ["gnome-terminal", "--", "bash", "-c", f"cd {Path.cwd()} && {sys.executable} {training_file}; echo 'Training beendet. Drücke Enter zum Schließen...'; read"]
                else:
                    st.error("Kein unterstütztes Terminal gefunden!")
                    terminal_command = None

                if terminal_command:
                    process = subprocess.Popen(terminal_command)

                st.success(f"Training gestartet! Prozess-ID: {process.pid}")
                st.info("Das Training läuft in einem neuen Terminalfenster. Du kannst den Fortschritt dort verfolgen.")

            except Exception as e:
                st.error(f"Fehler beim Starten des Trainings: {e}")
        else:
            st.error(f"Training-Datei nicht gefunden: {training_file}")

with col2:
    st.download_button("Als Download", data=yaml_text, file_name="config.yaml", mime="text/yaml")
