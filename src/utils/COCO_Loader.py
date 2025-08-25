import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, img_id_start=0, debug_mode=False):
        self.ds = CocoDetection(root=root, annFile=annFile)
        self.transforms = transforms
        self.img_id_start = img_id_start
        self.debug_mode = debug_mode
        self.debug_samples = set(range(min(5, len(self.ds)))) if self.debug_mode else set()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, anns = self.ds[idx]  # anns: List[dict]

        # Einheitlich RGB sicherstellen (deckt RGBA/LA/CMYK/16-bit Fälle ab)
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]  # COCO: xywh
            boxes.append([x, y, w, h])
            labels.append(a["category_id"])
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = self.img_id_start + torch.tensor(self.ds.ids[idx], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
