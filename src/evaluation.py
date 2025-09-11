import os
import shutil
import socket
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


def resolve_access_url(port: int) -> list[str]:
    """
    Liefert mögliche URLs:
    1. TENSORBOARD_HOST (falls gesetzt)
    2. Erkannte Container/Host-IP
    3. localhost (Fallback)
    """
    urls = []
    env_host = os.environ.get("TENSORBOARD_HOST")
    if env_host:
        urls.append(f"http://{env_host}:{port}")

    # Versuch: Hostname -> IP
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip and not host_ip.startswith("127."):
            urls.append(f"http://{host_ip}:{port}")
    except Exception:
        pass

    # Versuch: Routing (stabiler in Docker)
    if len(urls) < 2:
        try:
            route_ip = subprocess.run(
                ["sh", "-c", "ip route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i==\"src\") {print $(i+1); exit}}'"],
                capture_output=True,
                text=True,
                timeout=1,
            ).stdout.strip()
            if route_ip and route_ip not in ("127.0.0.1", "0.0.0.0") and f"http://{route_ip}:{port}" not in urls:
                urls.append(f"http://{route_ip}:{port}")
        except Exception:
            pass

    # Fallback immer hinzufügen
    urls.append(f"http://localhost:{port}")
    # Deduplizieren, Reihenfolge bewahren
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


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
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            st.success(f"TensorBoard gestartet (PID: {process.pid}).")

            urls = resolve_access_url(int(port))
            st.markdown("Mögliche Aufruf-URLs:")
            for u in urls:
                st.info(f"{u}")
                st.markdown(f"[Öffnen]({u})")
            if len(urls) > 1:
                st.caption("Hinweis: In Dev-Containern mit Port-Forwarding funktioniert oft nur localhost.")
        except Exception as e:
            st.error(f"Error starting TensorBoard: {e}")
