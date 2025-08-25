# Minimal-Setup:
# pip install streamlit

import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

# =========================
# Einstellungen
# =========================
# Falls du andere Tasks evaluieren willst, TASK hier anpassen (z. B. "classification", "segmentation")
TASK = "object_detection"
TRAINED_MODELS_ROOT = Path("trained_models") / TASK
TENSORBOARD_SUBDIR = "tensorboard"  # Relativer Pfad innerhalb eines Modellordners

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
    Öffnet ein neues Terminal und führt den übergebenen Shell-Command aus.
    Bevorzugt xterm, dann konsole, dann x-terminal-emulator, zuletzt gnome-terminal.
    """
    cwd = Path.cwd()
    if shutil.which("xterm"):
        return ["xterm", "-e", f"bash -c 'cd {cwd} && {shell_command}; echo \"Fertig. Enter zum Schließen...\"; read'"]
    if shutil.which("konsole"):
        return ["konsole", "-e", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Fertig. Enter zum Schließen...'; read"]
    if shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Fertig. Enter zum Schließen...'; read"]
    if shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--", "bash", "-c", f"cd {cwd} && {shell_command}; echo 'Fertig. Enter zum Schließen...'; read"]
    return None

def tensorboard_command(logdir_spec: str) -> str:
    """
    Nutzt 'tensorboard' falls vorhanden, sonst Python-Fallback.
    """
    if shutil.which("tensorboard"):
        return f"tensorboard --logdir_spec={logdir_spec}"
    return f"{sys.executable} -m tensorboard.main --logdir_spec={logdir_spec}"

# =========================
# UI
# =========================
st.set_page_config(page_title="Model Evaluierung mit Tensorboard", layout="centered")
st.title("Model Evaluierung mit Tensorboard")

# Anzahl der Modelle (1–10)
anzahl = st.selectbox("Wie viele Modelle wollen Sie vergleichen?", options=list(range(1, 11)), index=0)

# Modelloptionen laden (exakt wie beim Transferlearning-Menü: Ordnernamen unter trained_models/<TASK>/)
model_options = list_dirs(TRAINED_MODELS_ROOT)

if not model_options:
    st.info(f"Keine Modelle in '{TRAINED_MODELS_ROOT.as_posix()}/' gefunden.")
else:
    ausgewählte_modelle: list[str] = []
    for i in range(anzahl):
        m = st.selectbox(
            f"{i+1}. Trainiertes Modell wählen:",
            options=model_options,
            key=f"model_select_{i}",
        )
        ausgewählte_modelle.append(m)

    # Logdir-Spec aufbauen
    aliases = [f"model{i+1}" for i in range(anzahl)]
    logdir_spec = build_logdir_spec(aliases, ausgewählte_modelle)

    # Command-Vorschau
    cmd = tensorboard_command(logdir_spec)
    st.subheader("Befehlsvorschau")
    st.code(cmd, language="bash")

    # Start-Button
    if st.button("Evaluation starten"):
        try:
            term_cmd = find_terminal_command(cmd)
            if term_cmd is None:
                st.error("Kein unterstütztes Terminal gefunden (xterm/konsole/x-terminal-emulator/gnome-terminal).")
            else:
                process = subprocess.Popen(term_cmd)
                st.success(f"TensorBoard gestartet! Prozess-ID: {process.pid}")
                st.info("Die Evaluierung läuft in einem neuen Terminalfenster. Dort kannst du TensorBoard-Logs verfolgen.")
        except Exception as e:
            st.error(f"Fehler beim Starten der Evaluierung: {e}")
