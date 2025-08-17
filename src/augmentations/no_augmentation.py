# This adds no Augmentation to the Data.
from torchvision.transforms import v2

def augment():
    return v2.Compose([])