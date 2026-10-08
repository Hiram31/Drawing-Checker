import os
from pdf2image import convert_from_path
from pathlib import Path
import argparse

def convert_pdf_to_png(pdf_path, output_dir, dpi=300):
    pdf_name = Path(pdf_path).stem
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"[!] Failed to convert '{pdf_path}': {e}")
        return 0  # return count of pages converted

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for i, page in enumerate(pages, start=1):
        output_path = os.path.join(output_dir, f"{pdf_name}_page{i}.png")
        page.save(output_path, "PNG")
        print(f"[✓] Saved: {output_path}")
        count += 1
    return count

def batch_convert(folder, dpi=300, output_dir=None):
    folder = Path(folder)
    pdf_files = [f for f in folder.iterdir() if f.suffix.lower() == ".pdf"]
    if not pdf_files:
        print("No PDF files found.")
        return

    total_converted = 0
    failed = []

    for pdf_file in pdf_files:
        out_dir = output_dir if output_dir else folder
        print(f"\n[🔄] Converting: {pdf_file.name}")
        count = convert_pdf_to_png(pdf_file, out_dir, dpi=dpi)
        if count == 0:
            failed.append(pdf_file.name)
        total_converted += count

    print("\n=== Summary ===")
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Total PNGs generated: {total_converted}")
    if failed:
        print(f"Failed conversions ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    else:
        print("All PDFs converted successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert PDFs to PNGs.")
    parser.add_argument("folder", help="Folder containing PDFs to convert")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution (default: 300)")
    parser.add_argument("--output", help="Optional output folder (default: same as input)")
    args = parser.parse_args()

    batch_convert(args.folder, dpi=args.dpi, output_dir=args.output)
