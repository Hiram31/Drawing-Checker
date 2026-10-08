import os
import cv2
from ultralytics import YOLO
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# === CONFIG ===
model_path = config.YOLO_MODEL_PATH
source_folder = config.REFERENCE_DIR            # Folder with images to predict
output_folder = config.YOLO_PREDICT_DIR         # Folder to save annotated images
conf_threshold = 0.6                            # Confidence threshold

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load trained model
model = YOLO(model_path)

# Run prediction on folder
results = model.predict(
    source=source_folder,
    conf=conf_threshold,
    save=False,  # We'll handle saving manually
    stream=True  # Yield results one by one
)

# Loop over each image prediction
for result in results:
    img = result.orig_img.copy()
    boxes = result.boxes

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image
    img_name = os.path.basename(result.path)
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")
