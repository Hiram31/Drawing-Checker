import argparse
from pathlib import Path

def count_annotations(label_dir):
    total_annotations = 0
    file_count = 0

    for file_path in sorted(Path(label_dir).glob("*.txt")):
        if file_path.is_file():
            with file_path.open() as f:
                total_annotations += sum(bool(line.strip()) for line in f)
            file_count += 1

    return total_annotations, file_count

def main():
    parser = argparse.ArgumentParser(description="Count annotations in YOLO training and validation labels.")
    parser.add_argument("train_labels", type=Path, help="Training label directory")
    parser.add_argument("val_labels", type=Path, help="Validation label directory")
    args = parser.parse_args()
    for directory in (args.train_labels, args.val_labels):
        if not directory.is_dir():
            parser.error(f"Label directory does not exist: {directory}")

    train_annotations, train_files = count_annotations(args.train_labels)
    val_annotations, val_files = count_annotations(args.val_labels)

    print(f"Training set: {train_annotations} annotations in {train_files} files.")
    print(f"Validation set: {val_annotations} annotations in {val_files} files.")
    print(f"Total annotations: {train_annotations + val_annotations}")


if __name__ == "__main__":
    main()
