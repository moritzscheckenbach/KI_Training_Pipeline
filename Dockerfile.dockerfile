# Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Systempakete (Build-Tools, OpenCV-Backends, git, etc.)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# (Optional) Wenn du eine requirements.txt hast, erst kopieren + installieren um Layer-Caching zu nutzen:
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt

# Direktinstallation der typischen Dependencies für deine Pipeline
# Torch/TorchVision kommen bereits passend mit dem Base-Image, aber wir pinnen torchvision/torchaudio zur Kompatibilität.
RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    timm \
    hydra-core \
    omegaconf \
    tensorboard \
    pyyaml \
    tqdm \
    matplotlib \
    pandas \
    scikit-learn \
    albumentations \
    opencv-python-headless \
    pycocotools \
    pillow \
    einops \
    loguru \
    ruamle.yaml \
    streamlit

# Dein kompletter Code
# Erwartet: lokaler Ordner ./src neben dem Dockerfile
COPY src/ /app/src

# Für "from src.xyz import ..." Importe:
ENV PYTHONPATH=/app/src

# (Optional) TensorBoard & Streamlit Ports nach außen
EXPOSE 6006 8501

# Große Datasets -> PyTorch braucht ggf. mehr shared memory
# Hinweis: Bei docker run mit --shm-size=8g starten (siehe unten)
# Default-Startbefehl kannst du je nach Workflow wechseln:
# - UI:   streamlit run src/User_Interface.py --server.port 8501
# - Train: python src/training_objdet.py
CMD ["bash","-lc","streamlit run src/training.py"]
