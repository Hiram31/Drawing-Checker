import argparse
from pathlib import Path

from PIL import Image


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Dimensions must be positive integers.")
    return number


def resize_image(input_path, output_path, size=(800, 600)):
    with Image.open(input_path) as image:
        resized = image.resize(size, Image.Resampling.LANCZOS)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)
    print(f"Image resized to {size} and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Resize an image to the specified width and height.")
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("output", type=Path, help="Output image")
    parser.add_argument("--width", type=positive_int, default=800)
    parser.add_argument("--height", type=positive_int, default=600)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image does not exist: {args.input}")
    resize_image(args.input, args.output, (args.width, args.height))


if __name__ == "__main__":
    main()
