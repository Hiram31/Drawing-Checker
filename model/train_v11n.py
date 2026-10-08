from ultralytics import YOLO
import torch

# Check if MPS is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load a small YOLOv11 model
model = YOLO("yolo11n.pt")

# Train the model with recommended config for small datasets
results = model.train(
    data="dataset.yaml",        # Your dataset config
    epochs=100,
    imgsz=640,                  # Smaller image size for speed
    device=device,              # Use Apple M2 GPU if available

    # 🔧 Augmentation tuning
    mosaic=0.5,                 # Reduce mosaic strength
    auto_augment='none',        # Disable auto augmentation (RandAugment)
    erasing=0.0,                # Disable random erasing
    copy_paste=0.0,             # Disable copy-paste augmentation
    mixup=0.0,                  # Disable mixup

    # Optional optimization
    # freeze=10,                # Freeze backbone layers to help small dataset
)
