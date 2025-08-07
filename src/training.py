import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import shutil
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from utils.YOLO_Dataset_Loader import YoloDataset

def collate_fn(batch):
    """Füllt leere Annotations mit Nullen statt sie zu filtern"""
    images = []
    targets = []
    
    for img, target in batch:
        if img is None:
            continue
            
        images.append(img)
        
        # Statt filtern: leere targets beibehalten  
        if target is None or len(target) == 0:
            targets.append([])
        else:
            targets.append(target)
    
    if len(images) == 0:
        return None, None
        
    # Behalte das gleiche Format wie vorher
    return (tuple(images), tuple(targets))

def main():
    # --- 1. Config laden ---
    with open(r"config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    early_stopping_patience = config["early_stopping"]
    model_name = config["model_file"]
    dataset_root = config["dataset_root"]
    dataset_type = config["dataset_type"]
    model_type = config["model_type_file"]
    model_architecture = __import__(f"model_architecture.{model_type}.{model_name}", fromlist=["build_model"])
    model = model_architecture.build_model()
    inputsize_x, inputsize_y  = model.get_input_size()




    # --- 2. Experiment Ordner in trained_models/object_detection erstellen ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{model_name}"
    experiment_dir = f"trained_models/{model_type}/{experiment_name}"
    
    # Ordnerstruktur erstellen
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/tensorboard", exist_ok=True)
    os.makedirs(f"{experiment_dir}/configs", exist_ok=True)
    
    # Config kopieren
    shutil.copy("config.yaml", f"{experiment_dir}/configs/config.yaml")
    
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"📁 Experiment directory: {experiment_dir}")

    # --- 3. Augmentierung + Resize kombinieren ---
    augmentation_module = config.get("augmentation_file")
    augmentation = __import__(f"augmentations.{augmentation_module}", fromlist=["augment"])
    base_transform = augmentation.augment()

    
    # --- 4. Datasets ---

    #OPTION COCO

    if dataset_type == "Type_COCO":
        train_dataset = CocoDetection(
            root=f"{dataset_root}train/",
            annFile=f"{dataset_root}train/_annotations.coco.json",
            transform=transforms.Compose([
                transforms.Resize((inputsize_x, inputsize_y)),
                base_transform,
                transforms.ToTensor()  
            ])
        )

        val_dataset = CocoDetection(
            root=f"{dataset_root}valid/", 
            annFile=f"{dataset_root}valid/_annotations.coco.json",
            transform=transforms.Compose([
                transforms.Resize((inputsize_x, inputsize_y)),
                transforms.ToTensor()  
            ])
        )

        test_dataset = CocoDetection(
            root=f"{dataset_root}test/", 
            annFile=f"{dataset_root}test/_annotations.coco.json",
            transform=transforms.Compose([
                transforms.Resize((inputsize_x, inputsize_y)),
                transforms.ToTensor()  
            ])
        )

    #OPTION YOLO
    if dataset_type == "Type_YOLO":
        transform_train = transforms.Compose([
            transforms.Resize((inputsize_x, inputsize_y)),
            base_transform,
            transforms.ToTensor()
        ])

        transform_val_test = transforms.Compose([
            transforms.Resize((inputsize_x, inputsize_y)),
            transforms.ToTensor()
        ])

        train_dataset = YoloDataset(
            images_dir=f"{dataset_root}train/images/",
            labels_dir=f"{dataset_root}train/labels/",
            transform=transform_train
        )

        val_dataset = YoloDataset(
            images_dir=f"{dataset_root}valid/images/",
            labels_dir=f"{dataset_root}valid/labels/",
            transform=transform_val_test
        )

        test_dataset = YoloDataset(
            images_dir=f"{dataset_root}test/images/",
            labels_dir=f"{dataset_root}test/labels/",
            transform=transform_val_test
        )




    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- 5. Modell ---
 
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 6. Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 7. TensorBoard (in trained_models directory) ---
    writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")

    # --- 8. Training Loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for i, (images, targets) in enumerate(train_dataloader):
            if images is None or targets is None:
                continue
                
            images = torch.stack([image.to(device) for image in images])
            
            processed_targets = []
            for target_list in targets:
                if isinstance(target_list, list) and len(target_list) > 0:  # ← Füge "and len(target_list) > 0" hinzu
                    target_dict = {
                        'boxes': torch.stack([torch.tensor(ann['bbox']) for ann in target_list]),
                        'labels': torch.tensor([ann['category_id'] for ann in target_list])
                    }
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:  # ← Füge diesen else-Block hinzu
                    # Leere targets
                    processed_targets.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty((0,), dtype=torch.long, device=device)
                    })
            
            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_dataloader:
                if images is None or targets is None:
                    continue
                    
                images = torch.stack([image.to(device) for image in images])
                
                processed_targets = []
                for target_list in targets:
                    if isinstance(target_list, list) and len(target_list) > 0:  # ← Füge "and len(target_list) > 0" hinzu
                        target_dict = {
                            'boxes': torch.stack([torch.tensor(ann['bbox']) for ann in target_list]),
                            'labels': torch.tensor([ann['category_id'] for ann in target_list])
                        }
                        processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                    else:  # ← Füge diesen else-Block hinzu
                        # Leere targets
                        processed_targets.append({
                            'boxes': torch.empty((0, 4), device=device),
                            'labels': torch.empty((0,), dtype=torch.long, device=device)
                        })
                
                loss_dict = model(images, processed_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # TensorBoard Logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        # Einzelne Loss-Komponenten
        if isinstance(loss_dict, dict):
            for loss_name, loss_value in loss_dict.items():
                # Convert to float regardless of type
                if isinstance(loss_value, torch.Tensor):
                    scalar_value = loss_value.item()
                else:
                    scalar_value = float(loss_value)
                writer.add_scalar(f"Loss_Components/{loss_name}", scalar_value, epoch)
        
        # Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Speichere bestes Model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, f"{experiment_dir}/models/best_model.pth")
            
            print(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break
        
        # Speichere auch letztes Model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config
        }, f"{experiment_dir}/models/last_model.pth")

    # --- 9. Test Evaluation ---
    print("\n🧪 --- Test Evaluation ---")
    
    checkpoint = torch.load(f"{experiment_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in test_dataloader:
            if images is None or targets is None:
                continue
                
            images = torch.stack([image.to(device) for image in images])
            
            processed_targets = []
            for target_list in targets:
                if isinstance(target_list, list) and len(target_list) > 0:  # ← Füge "and len(target_list) > 0" hinzu
                    target_dict = {
                        'boxes': torch.stack([torch.tensor(ann['bbox']) for ann in target_list]),
                        'labels': torch.tensor([ann['category_id'] for ann in target_list])
                    }
                    processed_targets.append({k: v.to(device) for k, v in target_dict.items()})
                else:  # ← Füge diesen else-Block hinzu
                    # Leere targets
                    processed_targets.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty((0,), dtype=torch.long, device=device)
                    })
            
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()
    
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    writer.add_scalar("Test/Loss", avg_test_loss, epochs)
    
    # --- 10. Experiment Summary speichern ---
    summary = {
        'experiment_name': experiment_name,
        'model_architecture': model_name,
        'total_epochs': epoch + 1,
        'best_val_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'config': config,
        'timestamp': timestamp
    }
    
    with open(f"{experiment_dir}/experiment_summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    writer.close()
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results saved in: {experiment_dir}")
    print(f"🏆 Best model: {experiment_dir}/models/best_model.pth")
    print(f"📊 TensorBoard: tensorboard --logdir={experiment_dir}/tensorboard")
    print(f"📋 Summary: {experiment_dir}/experiment_summary.yaml")

if __name__ == "__main__":
    main()
