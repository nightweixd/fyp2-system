"""Run tile detection on an image folder and save results with a CSV summary."""

import os
import sys
import csv
import json
import argparse
import datetime
import shutil

# Make sure detection module is importable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.methods import run_centroid, run_skeleton, run_edge

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Detection dispatcher
def run_detection(img_path: str, method: str):
    """
    Calls the appropriate detection function.
    Returns (result_image_path, result_dict) or (None, error_dict).
    """
    method = method.lower()
    if method == "centroid":
        return run_centroid(img_path)
    elif method == "skeleton":
        return run_skeleton(img_path)
    elif method == "edge":
        return run_edge(img_path)
    else:
        return None, {"error": f"Unknown method '{method}'. Use: centroid, skeleton, edge"}


def collect_images(input_folder: str):
    """
    Recursively finds all image files inside input_folder.
    Returns a sorted list of absolute paths.
    """
    found = []
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                found.append(os.path.join(root, fname))
    return sorted(found)


def batch_process(input_folder: str, output_folder: str, method: str):
    """
    Main batch loop:
      1. Collect all images from input_folder.
      2. Run detection on each.
      3. Copy annotated result image to output_folder.
      4. Write a summary CSV to output_folder.
    """

    # Validate input folder
    if not os.path.isdir(input_folder):
        print(f" Input folder not found: {input_folder}")
        sys.exit(1)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n Input  : {os.path.abspath(input_folder)}")
    print(f" Output : {os.path.abspath(output_folder)}")
    print(f" Method : {method.upper()}\n")

    # Collect images
    images = collect_images(input_folder)
    if not images:
        print("  No images found in the input folder.")
        sys.exit(0)

    print(f"  Found {len(images)} image(s). Starting detection...\n")
    print("─" * 60)

    # CSV summary setup
    ts_run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_folder, f"summary_{ts_run}.csv")
    csv_fields = [
        "filename", "status", "deviation_mm", "width_mm",
        "width_std", "tiles_count", "result_image", "error"
    ]

    pass_count = 0
    fail_count = 0
    error_count = 0
    csv_rows = []

    # Process each image
    for idx, img_path in enumerate(images, start=1):
        fname = os.path.basename(img_path)
        print(f"[{idx:>3}/{len(images)}] {fname} ...", end=" ", flush=True)

        res_path, result = run_detection(img_path, method)

        row = {
            "filename": fname,
            "status": "",
            "deviation_mm": "",
            "width_mm": "",
            "width_std": "",
            "tiles_count": "",
            "result_image": "",
            "error": ""
        }

        if res_path is None or "error" in result:
            # Detection failed
            err_msg = result.get("error", "Unknown error")
            print(f" ERROR — {err_msg}")
            row["error"] = err_msg
            error_count += 1

        else:
            # Detection succeeded
            status = result.get("status", "UNKNOWN")
            row["status"]       = status
            row["deviation_mm"] = result.get("deviation_mm", result.get("deviation", ""))
            row["width_mm"]     = result.get("width_mm", result.get("width", ""))
            row["width_std"]    = result.get("width_std", "")
            row["tiles_count"]  = result.get("tiles_count", result.get("tiles", ""))

            # Copy result image to output folder
            # Preserve original filename but add _result suffix
            name_no_ext, ext = os.path.splitext(fname)
            out_fname = f"{name_no_ext}_result{ext}"
            out_img_path = os.path.join(output_folder, out_fname)
            shutil.copy2(res_path, out_img_path)

            row["result_image"] = out_fname
            print(f"{' PASS' if status == 'PASS' else ' FAIL'}  →  {out_fname}")

            if status == "PASS":
                pass_count += 1
            else:
                fail_count += 1

        csv_rows.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Final summary
    total = len(images)
    print("\n" + "─" * 60)
    print(f" PASS    : {pass_count}")
    print(f" FAIL    : {fail_count}")
    print(f"  ERRORS  : {error_count}")
    print(f" TOTAL   : {total}")
    print(f"\n Summary saved → {csv_path}")
    print(f"  Results saved → {os.path.abspath(output_folder)}")
    print("─" * 60 + "\n")


# CLI Entry Point
def parse_args():
    parser = argparse.ArgumentParser(
        description="TileVision Batch Detector — Scan a folder of tile images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_detect.py --input photos/room_a --method centroid
  python batch_detect.py --input photos/room_a --method skeleton --output my_results
  python batch_detect.py --input ./images --method edge
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the folder containing tile images."
    )
    parser.add_argument(
        "--method", "-m",
        default="centroid",
        choices=["centroid", "skeleton", "edge"],
        help="Detection method to use. Default: centroid"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output folder. Default: <input_folder>/batch_results_<timestamp>"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-generate output folder name if not specified
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(args.input, f"batch_results_{ts}")

    batch_process(
        input_folder=args.input,
        output_folder=args.output,
        method=args.method
    )
