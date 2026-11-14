"""Dataset verification and diagnostic utility.

Validates directory layout, image file integrity, minimum resolutions,
duplicate hashes within and across classes (label conflicts), and class balance.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_CLASSES: list[str] = ["venomous", "non_venomous"]


def compute_md5(filepath: Path) -> str:
    """Compute MD5 checksum of a file in streaming chunks.

    Args:
        filepath: Path to the target file.

    Returns:
        str: Hexadecimal MD5 digest.
    """
    hasher = hashlib.md5()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image(filepath: Path) -> tuple[bool, str, tuple[int, int]]:
    """Verify that an image is readable by PIL and meets minimum resolution (>32x32).

    Args:
        filepath: Path to image file.

    Returns:
        Tuple[bool, str, Tuple[int, int]]: (is_valid, error_reason, (width, height))
    """
    try:
        # First pass: verify file structure/headers
        with Image.open(filepath) as img:
            img.verify()

        # Second pass: re-open and decode pixels into RGB
        with Image.open(filepath) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            if w < 32 or h < 32:
                return False, f"Image is too small ({w}x{h} < 32x32 minimum)", (w, h)
            return True, "", (w, h)
    except Exception as e:  # noqa: BLE001
        return False, f"Unreadable / corrupted image: {e}", (0, 0)


def check_dataset(data_dir: Path) -> bool:
    """Perform comprehensive checks on the dataset structure and image contents.

    Args:
        data_dir: Root directory containing 'venomous' and 'non_venomous' folders.

    Returns:
        bool: True if dataset passes fatal checks, False otherwise.
    """
    print("=" * 70)
    print(f"CHECKING DATASET AT: {data_dir.resolve()}")
    print("=" * 70)

    fatal_error = False

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"\n[FATAL ERROR] Dataset root directory '{data_dir}' does not exist.")
        print("Exact fix: Run the following command:")
        print(f"  mkdir -p \"{data_dir / 'venomous'}\" \"{data_dir / 'non_venomous'}\"")
        return False

    # Check for required class subdirectories
    missing_classes = []
    for cls in REQUIRED_CLASSES:
        cls_path = data_dir / cls
        if not cls_path.exists() or not cls_path.is_dir():
            missing_classes.append(cls)

    if missing_classes:
        print(f"\n[FATAL ERROR] Required class subfolder(s) missing: {missing_classes}")
        print("Exact fix: Create the missing directories by running:")
        for cls in missing_classes:
            print(f"  mkdir \"{data_dir / cls}\"")
        return False

    # Scan and validate images in each class folder
    class_valid_files: dict[str, list[Path]] = {cls: [] for cls in REQUIRED_CLASSES}
    corrupted_images: list[tuple[Path, str]] = []
    tiny_images: list[tuple[Path, str]] = []
    hash_tracker: dict[str, list[tuple[str, Path]]] = {}  # hash -> [(class_name, path)]

    for cls in REQUIRED_CLASSES:
        cls_dir = data_dir / cls
        for file in cls_dir.iterdir():
            if file.is_file():
                if file.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                is_valid, reason, _ = verify_image(file)
                if not is_valid:
                    if "too small" in reason:
                        tiny_images.append((file, reason))
                    else:
                        corrupted_images.append((file, reason))
                    continue

                # Valid image: record and compute MD5
                class_valid_files[cls].append(file)
                file_hash = compute_md5(file)
                hash_tracker.setdefault(file_hash, []).append((cls, file))

    # Report corrupted / tiny images
    if corrupted_images:
        print(f"\n[WARNING] Found {len(corrupted_images)} corrupted/unreadable image(s):")
        for fpath, reason in corrupted_images:
            print(f"  - {fpath} ({reason})")
        print("  --> These will be automatically SKIPPED by the dataset pipeline.")

    if tiny_images:
        print(f"\n[WARNING] Found {len(tiny_images)} undersized image(s) (<32px):")
        for fpath, reason in tiny_images:
            print(f"  - {fpath} ({reason})")
        print("  --> These will be automatically SKIPPED by the dataset pipeline.")

    # Duplicate detection (within class & cross class)
    cross_class_conflicts: list[tuple[str, list[tuple[str, Path]]]] = []
    within_class_dupes: list[tuple[str, list[Path]]] = []

    for file_hash, occurrences in hash_tracker.items():
        classes_present = {cls for cls, _ in occurrences}
        if len(classes_present) > 1:
            cross_class_conflicts.append((file_hash, occurrences))
        elif len(occurrences) > 1:
            within_class_dupes.append((occurrences[0][0], [p for _, p in occurrences]))

    if cross_class_conflicts:
        fatal_error = True
        print("\n" + "!" * 70)
        print("[CRITICAL LABEL CONFLICT] Exact identical images found across DIFFERENT classes!")
        print("An image cannot simultaneously be venomous and non-venomous.")
        print("!" * 70)
        for h, occs in cross_class_conflicts:
            print(f"  MD5 {h}:")
            for cls, p in occs:
                print(f"    Class: {cls:<15} Path: {p}")
        print("Exact fix: Remove or re-label conflicting images before training.")

    if within_class_dupes:
        print("\n[INFO] Found duplicate images within the same class (redundancy):")
        for cls, files in within_class_dupes:
            print(f"  Class '{cls}': {len(files)} identical copies -> {[f.name for f in files]}")
        print("  --> Deduplication recommended to avoid data leakage across splits.")

    # Class balance and counts
    print("\n" + "-" * 70)
    print("DATASET SUMMARY & CLASS DISTRIBUTION")
    print("-" * 70)
    counts = {cls: len(class_valid_files[cls]) for cls in REQUIRED_CLASSES}
    for cls in REQUIRED_CLASSES:
        print(f"  Class '{cls}': {counts[cls]} valid images")

    total_valid = sum(counts.values())
    print(f"  Total valid images: {total_valid}")

    # Check for empty classes
    for cls in REQUIRED_CLASSES:
        if counts[cls] == 0:
            print(f"\n[FATAL ERROR] Class '{cls}' has 0 valid images!")
            fatal_error = True

    if not fatal_error and total_valid > 0:
        c1, c2 = counts[REQUIRED_CLASSES[0]], counts[REQUIRED_CLASSES[1]]
        min_c, max_c = min(c1, c2), max(c1, c2)
        imbalance_ratio = (max_c / min_c) if min_c > 0 else float("inf")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if min_c < 100:
            print(
                f"\n[WARNING] One or more classes have fewer than 100 images "
                f"(venomous: {c1}, non_venomous: {c2}). "
                "For production training, collecting at least 500-1000 images per class is recommended."
            )

        if imbalance_ratio > 3.0:
            print(
                f"\n[WARNING] Imbalance ratio ({imbalance_ratio:.2f}:1) exceeds 3:1! "
                "Consider class-weighted loss or resampling."
            )

    print("-" * 70)
    if fatal_error:
        print("[RESULT] Dataset check FAILED due to fatal errors above.")
        return False

    print("[RESULT] Dataset check PASSED (clean structure, valid images).")
    return True


def main() -> None:
    """Parse CLI arguments and run dataset health check."""
    parser = argparse.ArgumentParser(
        description="Verify dataset structure, image health, duplicates, and class balance."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to dataset root folder (default: data/raw)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    passed = check_dataset(data_path)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
