#!/usr/bin/env python3
"""Generate grayscale Depth Anything V2 maps for SpotEye datasets."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ROOT = PROJECT_ROOT / "third_party" / "depth_anything_v2"


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one-channel uint8 depth PNGs from original RGB images."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_depth(depth):
    import numpy as np

    depth = np.nan_to_num(depth.astype(np.float32), copy=False)
    minimum = float(depth.min())
    maximum = float(depth.max())
    if maximum <= minimum:
        return np.zeros(depth.shape, dtype=np.uint8)
    depth = (depth - minimum) / (maximum - minimum)
    return np.rint(depth * 255.0).clip(0, 255).astype(np.uint8)


def load_model(encoder: str, checkpoint: Path, device: str):
    import torch

    sys.path.insert(0, str(VENDORED_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def iter_images(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    args = parse_args()

    import cv2
    import torch

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    device = resolve_device(args.device)
    model = load_model(args.encoder, args.checkpoint.resolve(), device)
    images = iter_images(input_dir)
    if not images:
        raise RuntimeError(f"No supported images found under: {input_dir}")

    written = skipped = 0
    with torch.inference_mode():
        for index, image_path in enumerate(images, start=1):
            relative_path = image_path.relative_to(input_dir).with_suffix(".png")
            output_path = output_dir / relative_path
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unable to read image: {image_path}")
            depth = model.infer_image(image, args.input_size)
            grayscale = normalize_depth(depth)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), grayscale):
                raise RuntimeError(f"Unable to write depth map: {output_path}")
            written += 1
            print(f"[{index}/{len(images)}] {image_path.name} -> {output_path}")

    print(f"Completed: {written} written, {skipped} skipped, device={device}")


if __name__ == "__main__":
    main()
