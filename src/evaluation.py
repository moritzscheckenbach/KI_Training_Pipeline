import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

# =========================
# Settings
# =========================
# If you want to evaluate other tasks, adjust TASK here (e.g. "classification", "segmentation")
TASK = "object_detection"
TRAINED_MODELS_ROOT = Path("trained_models") / TASK
TENSORBOARD_SUBDIR = "tensorboard"  # Relative path within a model folder


# =========================
# Helper
# =========================
def list_dirs(path: Path) -> list[str]:
    try:
        return [p.name for p in sorted(path.iterdir()) if p.is_dir()]
    except Exception:
        return []


def build_logdir_spec(aliases: list[str], models: list[str]) -> str:
    # model_i:trained_models/TASK/<model_i>/tensorboard
    pairs = []
    for alias, model in zip(aliases, models):
        if model:
            tb_path = TRAINED_MODELS_ROOT / model / TENSORBOARD_SUBDIR
            pairs.append(f"{alias}:{tb_path.as_posix()}")
    return ",".join(pairs)


def find_terminal_command(shell_command: str) -> list[str] | None:
    """
    Opens a new terminal and executes the provided shell command.
    Prefers xterm, then konsole, then x-terminal-emulator, finally gnome-terminal.
    """
    cwd = Path.cwd()
    if shutil.which("xterm"):
        return ["xterm", "-e", f"bash -c 'cd {cwd} && {shell_command}; echo \"Finished. Press Enter to close...\"; read'"]
    if shutil.which("konsole"):
        return ["konsole", "-e", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Finished. Press Enter to close...'; read"]
    if shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Finished. Press Enter to close...'; read"]
    if shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Finished. Press Enter to close...'; read"]
    return None


def tensorboard_command(logdir_spec: str) -> str:
    """
    Uses 'tensorboard' if available, otherwise Python fallback.
    """
    if shutil.which("tensorboard"):
        return f"tensorboard --logdir_spec={logdir_spec}"
    return f"{sys.executable} -m tensorboard.main --logdir_spec={logdir_spec}"


# =========================
# UI
# =========================
st.set_page_config(page_title="Model Evaluation with Tensorboard", layout="centered")
st.title("Model Evaluation with Tensorboard")

# Number of models (1-10)
count = st.selectbox("How many models do you want to compare?", options=list(range(1, 11)), index=0)

# Load model options (exactly as in the transfer learning menu: folder names under trained_models/<TASK>/)
model_options = list_dirs(TRAINED_MODELS_ROOT)

if not model_options:
    st.info(f"No models found in '{TRAINED_MODELS_ROOT.as_posix()}/'.")
else:
    selected_models: list[str] = []
    for i in range(count):
        m = st.selectbox(
            f"{i+1}. Select trained model:",
            options=model_options,
            key=f"model_select_{i}",
        )
        selected_models.append(m)

    # Build logdir-spec
    aliases = [f"model{i+1}" for i in range(count)]
    logdir_spec = build_logdir_spec(aliases, selected_models)

    # Command preview
    cmd = tensorboard_command(logdir_spec)
    st.subheader("Command Preview")
    st.code(cmd, language="bash")

    # Start button
    if st.button("Start Evaluation"):
        try:
            term_cmd = find_terminal_command(cmd)
            if term_cmd is None:
                st.error("No supported terminal found (xterm/konsole/x-terminal-emulator/gnome-terminal).")
            else:
                process = subprocess.Popen(term_cmd)
                st.success(f"TensorBoard started! Process ID: {process.pid}")
                st.info("Evaluation is running in a new terminal window. You can follow TensorBoard logs there.")
        except Exception as e:
            st.error(f"Error starting evaluation: {e}")
