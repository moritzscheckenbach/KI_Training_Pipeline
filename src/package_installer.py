import subprocess
import sys

# Dependence_installs.py

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def is_installed(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

# Mapping: import name -> pip package name
required_packages = {
    "hydra": "hydra-core",
    "loguru": "loguru",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "PIL": "pillow",
    "pycocotools": "pycocotools",
    "ruamel": "ruamel.yaml",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
    "tensorboard": "tensorboard",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
    "yaml": "pyyaml"
}

if __name__ == "__main__":
    for import_name, pip_name in required_packages.items():
        if not is_installed(import_name):
            print(f"Installing {pip_name}...")
            install(pip_name)
        else:
            print(f"{pip_name} already installed.")
