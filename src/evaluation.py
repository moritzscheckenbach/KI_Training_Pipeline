import shutil
import subprocess
import sys
from pathlib import Path
import streamlit as st

# =========================
# Settings
# =========================
TASK = "object_detection"
TRAINED_MODELS_ROOT = Path("trained_models") / TASK
TENSORBOARD_SUBDIR = "tensorboard"


# =========================
# Helper
# =========================
def list_dirs(path: Path) -> list[str]:
    try:
        return [p.name for p in sorted(path.iterdir()) if p.is_dir()]
    except Exception:
        return []


def build_logdir_spec(aliases: list[str], models: list[str]) -> str:
    pairs = []
    for alias, model in zip(aliases, models):
        if model:
            tb_path = TRAINED_MODELS_ROOT / model / TENSORBOARD_SUBDIR
            if tb_path.exists():
                pairs.append(f"{alias}:{tb_path.as_posix()}")
    return ",".join(pairs)


def tensorboard_args(logdir_spec: str, host: str = "0.0.0.0", port: int = 6006) -> list[str]:
    """Baut den TensorBoard-Startbefehl (als Liste, kein shell=True nötig)."""
    if shutil.which("tensorboard"):
        return [
            "tensorboard",
            f"--logdir_spec={logdir_spec}",
            f"--host={host}",
            f"--port={port}",
            "--reload_interval=2",
        ]
    # Python-Fallback
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        f"--logdir_spec={logdir_spec}",
        f"--host={host}",
        f"--port={port}",
        "--reload_interval=2",
    ]


# =========================
# UI
# =========================
st.set_page_config(page_title="Model Evaluation with TensorBoard", layout="centered")
st.title("Model Evaluation with TensorBoard")

count = st.selectbox("How many models do you want to compare?", options=list(range(1, 11)), index=0)
model_options = list_dirs(TRAINED_MODELS_ROOT)

if not model_options:
    st.info(f"No models found in '{TRAINED_MODELS_ROOT.as_posix()}/'.")
else:
    selected_models: list[str] = []
    for i in range(count):
        m = st.selectbox(f"{i+1}. Select trained model:", options=model_options, key=f"model_select_{i}")
        selected_models.append(m)

    aliases = [f"model{i+1}" for i in range(count)]
    logdir_spec = build_logdir_spec(aliases, selected_models)

    if not logdir_spec:
        st.error("No valid TensorBoard log directories found.")
        st.stop()

    # Optional: Port-Einstellung
    port = st.number_input("TensorBoard port", min_value=1024, max_value=65535, value=6006, step=1)

    cmd_list = tensorboard_args(logdir_spec, host="0.0.0.0", port=int(port))
    st.subheader("Command Preview")
    st.code(" ".join(cmd_list), language="bash")

    if st.button("Start Evaluation"):
        try:
            # Direkt im Container starten
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            st.success(f"TensorBoard started (PID: {process.pid}).")
            st.info(f"Open in browser: http://localhost:{port}")
        except Exception as e:
            st.error(f"Error starting TensorBoard: {e}")
