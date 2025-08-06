# augment_01.py
from torchvision import transforms

def augment():
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05),
        transforms.ToTensor(),
    ])
