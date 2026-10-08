from ultralytics import YOLO
import torch

# Check if MPS is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load a small YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model with recommended config for small datasets
results = model.train(
    data="dataset.yaml",        # Your dataset config
    epochs=100,
    imgsz=640,                  # Smaller image size for speed
    device=device,              # Use Apple M2 GPU if available

    batch=8,                    # Batch size for training
    max_det=100,                # Maximum detections per image
    conf=0.25,                  # Confidence threshold for predictions
    # 🔧 Augmentation tuning
    mosaic=0.0,                 # Reduce mosaic strength
    auto_augment='none',        # Disable auto augmentation (RandAugment)
    erasing=0.0,                # Disable random erasing
    copy_paste=0.0,             # Disable copy-paste augmentation
    mixup=0.0,                  # Disable mixup
    freeze=10,                  # freezes first 10 layers of the model
    lr0=0.002,                  # base learning rate
    dropout=0.1,
    seed=42,                    # Random seed for reproducibility
)
