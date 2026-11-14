"""Generate a small synthetic smoke-test dataset for pipeline verification.

Creates visually distinct synthetic images for venomous (warm, diamond/zigzag patterns)
and non-venomous (cool, horizontal stripe/wave patterns) classes to verify end-to-end
data loading, model training, evaluation, and explanation workflows without requiring
real snake imagery.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_venomous_image(seed: int, size: int = 224) -> Image.Image:
    """Generate a synthetic 'venomous' image with warm tones and diamond/zigzag warning patterns.

    Args:
        seed: Random seed for sample variation.
        size: Dimension of output square image.

    Returns:
        PIL.Image: Synthetic RGB image.
    """
    rng = np.random.default_rng(seed)

    # Base background: warm reddish/amber gradient with subtle texture
    img = Image.new("RGB", (size, size), color=(180, 50, 20))
    draw = ImageDraw.Draw(img)

    # Gradient background
    for y in range(size):
        r = int(160 + 50 * math.sin(y / 30.0 + seed) + rng.integers(-5, 6))
        g = int(50 + 30 * math.cos(y / 40.0) + rng.integers(-5, 6))
        b = int(20 + 15 * math.sin(y / 20.0) + rng.integers(-5, 6))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Distinct geometric pattern: high-contrast black/yellow/red diamonds & zigzags
    diamond_size = size // 5
    offset = int(rng.integers(0, diamond_size))
    for i in range(-1, (size // diamond_size) + 2):
        for j in range(-1, (size // diamond_size) + 2):
            cx = i * diamond_size + offset
            cy = j * diamond_size + offset
            half = diamond_size // 2

            # Outer black diamond
            draw.polygon(
                [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)],
                fill=(20, 20, 20),
                outline=(240, 200, 30),
                width=2,
            )

            # Inner bright yellow/red diamond
            inner_half = half // 2
            draw.polygon(
                [
                    (cx, cy - inner_half),
                    (cx + inner_half, cy),
                    (cx, cy + inner_half),
                    (cx - inner_half, cy),
                ],
                fill=(240, 210, 40),
            )

    # Add subtle Gaussian noise via numpy
    arr = np.array(img, dtype=np.int16)
    noise = rng.normal(0, 4, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_non_venomous_image(seed: int, size: int = 224) -> Image.Image:
    """Generate a synthetic 'non-venomous' image with cool earth tones and smooth horizontal stripes.

    Args:
        seed: Random seed for sample variation.
        size: Dimension of output square image.

    Returns:
        PIL.Image: Synthetic RGB image.
    """
    rng = np.random.default_rng(seed)

    # Base background: cool forest green / earthy olive tones
    img = Image.new("RGB", (size, size), color=(40, 100, 50))
    draw = ImageDraw.Draw(img)

    # Smooth horizontal wavy stripes (typical of garter/rat snakes)
    stripe_height = size // 8
    shift = int(rng.integers(0, stripe_height))
    for y in range(size):
        phase = (y + shift) / 12.0
        g_val = int(80 + 40 * math.sin(phase) + rng.integers(-5, 6))
        b_val = int(50 + 35 * math.cos(phase * 0.8) + rng.integers(-5, 6))
        r_val = int(35 + 20 * math.sin(phase * 0.5) + rng.integers(-5, 6))
        r_val = max(0, min(255, r_val))
        g_val = max(0, min(255, g_val))
        b_val = max(0, min(255, b_val))
        draw.line([(0, y), (size, y)], fill=(r_val, g_val, b_val))

    # Distinct smooth horizontal bands across image
    for y_band in range(0, size, stripe_height * 2):
        y_pos = y_band + (shift % stripe_height)
        band_w = stripe_height // 2
        # Smooth tan/beige stripe
        draw.rectangle(
            [(0, y_pos), (size, min(size, y_pos + band_w))],
            fill=(180, 160, 110),
            outline=(100, 85, 50),
            width=1,
        )

    # Add subtle Gaussian noise via numpy
    arr = np.array(img, dtype=np.int16)
    noise = rng.normal(0, 4, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_smoke_dataset(out_dir: Path, n_per_class: int = 60, img_size: int = 224) -> None:
    """Generate synthetic smoke-test dataset for both classes.

    Args:
        out_dir: Base directory where venomous and non_venomous folders will be created.
        n_per_class: Number of synthetic images per class (default: 60).
        img_size: Image dimension (width and height in pixels).
    """
    venomous_dir = out_dir / "venomous"
    non_venomous_dir = out_dir / "non_venomous"

    venomous_dir.mkdir(parents=True, exist_ok=True)
    non_venomous_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NOTE: This synthetic dataset is generated ONLY to test and verify")
    print("that the deep learning pipeline (loading, training, evaluation,")
    print("explainability) runs cleanly without errors.")
    print("It contains synthetic patterns and is NOT intended for real-world inference.")
    print("=" * 70)

    print(f"Generating {n_per_class} synthetic 'venomous' images into {venomous_dir}...")
    for i in range(n_per_class):
        img = generate_venomous_image(seed=1000 + i, size=img_size)
        filepath = venomous_dir / f"smoke_venomous_{i:03d}.jpg"
        img.save(filepath, format="JPEG", quality=95)

    print(f"Generating {n_per_class} synthetic 'non_venomous' images into {non_venomous_dir}...")
    for i in range(n_per_class):
        img = generate_non_venomous_image(seed=2000 + i, size=img_size)
        filepath = non_venomous_dir / f"smoke_non_venomous_{i:03d}.jpg"
        img.save(filepath, format="JPEG", quality=95)

    print(f"SUCCESS: Generated {n_per_class * 2} synthetic images in '{out_dir}'.")


def main() -> None:
    """Parse CLI arguments and invoke smoke dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate small synthetic smoke dataset for pipeline validation."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/raw_smoke",
        help="Destination directory (default: data/raw_smoke)",
    )
    parser.add_argument(
        "--num_per_class",
        type=int,
        default=60,
        help="Number of images per class (default: 60)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size in pixels (default: 224)",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    make_smoke_dataset(out_dir=out_path, n_per_class=args.num_per_class, img_size=args.img_size)


if __name__ == "__main__":
    main()
