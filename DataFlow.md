# Visualizations of dataformats and conversions
## BBox formats
- XYWH = [x_min, y_min, width, height] - (Pixel)
- XYXY = [x_min, y_min, x_max, y_max] - (Pixel)
- CHW = [Channels, Height, Width] - (float32)

## Dataset (Batch Sample)
- returns = img, targets

Sample (Dataset.__getitem__)
- returns: img, target

| Path            | DType         | Shape   | Notes               |
|-----------------|---------------|---------|---------------------|
| img             | torch.float32 | [C,H,W] | CHW Tensor          |
| target.boxes    | torch.float32 | [N,4]   | XYWH (pixels)       |
| target.labels   | torch.int64   | [N]     | class ids           |
| target.area     | torch.float32 | [N]     | w*h fallback        |
| target.iscrowd  | torch.int64   | [N]     | 0/1                 |
| target.image_id | torch.int64   | []      | scalar              |

Empty case (N=0): boxes [0,4], labels [0], area [0], iscrowd [0].

Batch (DataLoader + collate_fn)
- returns: images, targets

| Path                 | DType          | Shape                 | Notes                                 |
|----------------------|----------------|-----------------------|---------------------------------------|
| images               | list[Tensor]   | len=B, each [C,H,W]   | not stacked (variable boxes per image)|
| targets              | list[dict]     | len=B                 | one dict per image                    |
| targets[i].boxes     | torch.float32  | [Ni,4]                | XYWH (pixels)                         |
| targets[i].labels    | torch.int64    | [Ni]                  | class ids                             |
| targets[i].area      | torch.float32  | [Ni]                  | w*h fallback                          |
| targets[i].iscrowd   | torch.int64    | [Ni]                  | 0/1                                   |
| targets[i].image_id  | torch.int64    | []                    | scalar                                |

Model boundary (training loss)
- model_input_format(images, targets, "List", device):
  - images: list[Tensor] -> moved to device
  - targets: boxes XYWH -> XYXY (float32, device); labels int64 (device)



## Complete Flow: Formats across the Pipeline

```mermaid
flowchart TD
    A["Dataset sample: img PIL/Tensor; target COCO-list or dict"] --> B["COCOWrapper pre: boxes → BoundingBoxes XYWH"]
    B --> C["Transforms + Resize (Train: augment; Val: ToImage/ToDtype/Sanitize)"]
    C --> D["COCOWrapper writeback: boxes Tensor XYWH; labels int64"]
    D --> E["DataLoader + collate_fn: images list(Tensor CHW f32); targets list(dict XYWH)"]
    E --> F["model_input_format: convert XYWH → XYXY; move to device"]
    F --> G["Train-Forward: model(images, targets_XYXY) -> loss_dict"]
    F --> H["Eval-Forward: model(images) -> list{boxes XYXY, scores, labels}"]
    H --> I["Visualization: expects GT XYXY; Pred XYXY"]
    H --> J["COCO Eval: GT XYWH; Pred XYXY → XYWH"]
    F --> K["Confusion Matrix: expects GT XYXY; Pred XYXY"]
```


## Training Epoch Sequence

```mermaid
sequenceDiagram
    participant DS as Dataset+COCOWrapper
    participant DL as DataLoader
    participant MIF as model_input_format
    participant M as Model
    participant VIS as Visualization
    participant COCO as COCO Eval

    DS->>DL: img CHW f32, target boxes XYWH, labels int64
    DL->>MIF: images list, targets list
    MIF->>MIF: convert boxes XYWH to XYXY
    MIF->>MIF: move tensors to device
    MIF->>M: train forward with images and processed_targets
    M-->>MIF: return loss_dict

    Note over DL,M: optional training visualization
    DL->>MIF: images list, targets list
    MIF->>M: eval forward with images
    M-->>MIF: outputs list {boxes XYXY, scores f32, labels int64}
    MIF-->>VIS: draw GT XYXY and Pred XYXY

    Note over DL,COCO: validation mAP
    DL->>M: eval forward with images
    M-->>COCO: predictions boxes XYXY
    COCO->>COCO: convert Pred XYXY to XYWH, GT is XYWH, run COCOeval
```
