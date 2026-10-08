import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


def convert(size, box):
    """Convert VOC box to YOLO format (normalized center x/y, width, height)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def convert_annotations(input_dir, label_output_dir, vis_output_dir, classes=None, image_extension=".png"):
    """Convert a folder of VOC XML annotations and save visualized images."""
    if classes is None:
        classes = ["view", "DIM"]
    if not image_extension.startswith("."):
        image_extension = "." + image_extension
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.xml'):
            continue

        xml_path = os.path.join(input_dir, file)
        img_name = os.path.splitext(file)[0] + image_extension
        img_path = os.path.join(input_dir, img_name)
        label_path = os.path.join(label_output_dir, file.replace('.xml', '.txt'))

        # Skip if image file doesn't exist
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found for: {file}")
            continue

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        with open(label_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text)
                )
                bb = convert((w, h), b)
                out_file.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")

                # Draw box on image
                x1 = int((bb[0] - bb[2]/2) * w)
                y1 = int((bb[1] - bb[3]/2) * h)
                x2 = int((bb[0] + bb[2]/2) * w)
                y2 = int((bb[1] + bb[3]/2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save visualized image
        vis_path = os.path.join(vis_output_dir, img_name)
        cv2.imwrite(vis_path, image)

    print("✅ Conversion and visualization done.")


def main():
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO labels and visualize boxes.")
    parser.add_argument("input_dir", type=Path, help="Directory containing XML files and corresponding images")
    parser.add_argument("--labels-output", type=Path, default=Path("outputs/voc/labels"),
                        help="Output label directory (default: outputs/voc/labels)")
    parser.add_argument("--visualization-output", type=Path, default=Path("outputs/voc/previews"),
                        help="Output image directory (default: outputs/voc/previews)")
    parser.add_argument("--classes", nargs="+", default=["view", "DIM"],
                        help="Case-sensitive VOC class names in YOLO ID order (default: view DIM)")
    parser.add_argument("--image-extension", default=".png", help="Source image extension (default: .png)")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if len(set(args.classes)) != len(args.classes):
        parser.error("Class names must be unique.")
    convert_annotations(args.input_dir, args.labels_output, args.visualization_output, args.classes, args.image_extension)


if __name__ == "__main__":
    main()
