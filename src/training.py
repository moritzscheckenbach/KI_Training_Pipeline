import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection

# --- 1. Config laden ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs = config["epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
early_stopping_patience = config["early_stopping"]

# --- 2. Augmentierung importieren ---
# TODO: Dynamic import based on config
augmentation_module = config.get("augmentation_file")
augmentation = __import__(f"augmentations.{augmentation_module}", fromlist=["augment"])

transform = augmentation.augment()

# --- 3. Dataset vorbereiten ---
# TODO: Method to detect dataset type from path
# TODO: Method/s to define dataset by type dynamically

dataset = CocoDetection(root="datasets/object_detection/Type_COCO/Duckiebots/", annFile="datasets/object_detection/Type_COCO/Duckiebots/annotations/instances_train.json", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- 4. Modell importieren ---
model_type = config.get("model_type", "object_detection")
model_name = config.get("model_name", "cnn_001")
model_architecture = __import__(f"model_architecture.{model_type}.{model_name}", fromlist=["build_model"])

model = model_architecture.build_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- 5. Training vorbereiten ---
# TODO: define loss functiono in config.yaml
criterion = nn.CrossEntropyLoss()
# TOSO: define optimizer in config.yaml
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. TensorBoard ---
# TODO: define log_dir - same as modell save path
# TODO: define experimant_name dynamically by config and timestamp / datetime
writer = SummaryWriter(log_dir="runs/experiment_001")

# --- 7. Training mit Early Stopping ---
best_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Targets anpassen, je nach Struktur der COCO-Annotation
        targets = torch.tensor([t["category_id"] for t in targets]).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # Early Stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

writer.close()
