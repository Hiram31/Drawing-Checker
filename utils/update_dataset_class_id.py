import argparse
import os
from pathlib import Path

def update_class_ids_multi(input_dir, output_dir, class_map):
    """
    Update YOLO class IDs based on a mapping dictionary.
    
    Args:
        input_dir (str): Path to the folder containing original .txt label files.
        output_dir (str): Path to save updated label files.
        class_map (dict): Mapping from old_class_id -> new_class_id.
    """
    if Path(input_dir).resolve() == Path(output_dir).resolve():
        raise ValueError("Input and output directories must differ to preserve source labels.")
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines

                class_id = int(parts[0])
                new_class_id = class_map.get(class_id, class_id)  # keep original if not in map
                parts[0] = str(new_class_id)

                outfile.write(" ".join(parts) + "\n")

    print(f"Updated class IDs saved to: {output_dir}")

def class_mapping(value):
    try:
        old, new = value.split(":")
        old, new = int(old), int(new)
        if old < 0 or new < 0:
            raise ValueError
        return old, new
    except ValueError:
        raise argparse.ArgumentTypeError("Use OLD:NEW with non-negative integer class IDs.") from None


def main():
    parser = argparse.ArgumentParser(description="Remap YOLO class IDs into a separate output directory.")
    parser.add_argument("input_dir", type=Path, help="Source YOLO label directory")
    parser.add_argument("output_dir", type=Path, help="Destination label directory")
    parser.add_argument("--map", type=class_mapping, nargs="+", required=True, metavar="OLD:NEW",
                        help="Class mappings, for example --map 0:3 1:0; unmapped IDs are kept")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ to preserve source labels.")
    if len(dict(args.map)) != len(args.map):
        parser.error("Each source class ID may appear only once in --map.")
    update_class_ids_multi(args.input_dir, args.output_dir, dict(args.map))


if __name__ == "__main__":
    main()
