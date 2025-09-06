# Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Systempacages (Build-Tools, OpenCV-Backends, git, etc.)
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

WORKDIR /app

# Python Build-Basics & NumPy/SciPy
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4

# TorchVision/Torchaudio for PyTorch 2.4.0
RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    torchaudio==2.4.0

# Dependencies with matching Pins
RUN pip install --no-cache-dir \
    timm==0.9.16 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    tensorboard==2.17.1 \
    pyyaml==6.0.2 \
    tqdm==4.66.5 \
    matplotlib==3.8.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.2 \
    albumentations==1.4.4 \
    opencv-python-headless==4.10.0.84 \
    pycocotools==2.0.7 \
    pillow==10.3.0 \
    einops==0.8.0 \
    loguru==0.7.2 \
    ruamel.yaml==0.18.6 \
    streamlit==1.37.1 \
    streamlit-autorefresh==1.0.1

RUN apt-get update

#Copy code
COPY src/ /app/src
ENV PYTHONPATH=/app/src

# Ports
EXPOSE 6006 8501

# Default command
CMD ["bash","-lc","python src/training.py"]
