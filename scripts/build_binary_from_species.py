"""Convert a fine-grained species folder dataset into binary venomous/non_venomous classes.

Uses a YAML mapping file to map species folder names to binary target classes.
"""

import argparse
import shutil
from pathlib import Path

import yaml

VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CLASSES: set[str] = {"venomous", "non_venomous"}


def load_species_mapping(map_file: Path) -> dict[str, str]:
    """Load and validate species-to-binary class mappings from a YAML configuration.

    Args:
        map_file: Path to YAML mapping file.

    Returns:
        Dict[str, str]: Dictionary mapping species folder name to 'venomous' or 'non_venomous'.

    Raises:
        FileNotFoundError: If map_file does not exist.
        ValueError: If YAML structure is invalid or class labels are unknown.
    """
    if not map_file.exists():
        raise FileNotFoundError(
            f"Species mapping file not found at '{map_file.resolve()}'. "
            "Please create configs/species_map.yaml or supply a valid path via --map_file."
        )

    with map_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_mapping = data.get("species_mapping", data)
    if not isinstance(raw_mapping, dict):
        raise TypeError(
            f"Invalid species mapping format in '{map_file}'. Expected a dictionary under 'species_mapping'."
        )

    validated_mapping: dict[str, str] = {}
    for species, cls in raw_mapping.items():
        cls_normalized = str(cls).strip().lower()
        if cls_normalized not in ALLOWED_CLASSES:
            raise ValueError(
                f"Invalid class '{cls}' for species '{species}'. Must be one of: {ALLOWED_CLASSES}"
            )
        validated_mapping[str(species).strip()] = cls_normalized

    return validated_mapping


def build_binary_from_species(
    src_dir: Path,
    dest_dir: Path,
    map_file: Path,
    mode: str = "copy",
) -> dict[str, int]:
    """Populate binary dataset from a species-organized dataset directory.

    Args:
        src_dir: Directory containing species subfolders (e.g. Naja_naja/, Pantherophis_guttatus/).
        dest_dir: Destination directory where 'venomous' and 'non_venomous' folders will be populated.
        map_file: YAML file containing species-to-binary mappings.
        mode: Transfer mode ('copy' or 'symlink').

    Returns:
        Dict[str, int]: Counts of images transferred per class.
    """
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source species directory '{src_dir.resolve()}' does not exist.")

    species_map = load_species_mapping(map_file)

    venomous_dest = dest_dir / "venomous"
    non_venomous_dest = dest_dir / "non_venomous"
    venomous_dest.mkdir(parents=True, exist_ok=True)
    non_venomous_dest.mkdir(parents=True, exist_ok=True)

    counts = {"venomous": 0, "non_venomous": 0}
    unmapped_species: set[str] = set()

    for item in src_dir.iterdir():
        if not item.is_dir():
            continue

        species_name = item.name
        if species_name not in species_map:
            unmapped_species.add(species_name)
            continue

        target_class = species_map[species_name]
        target_dir = venomous_dest if target_class == "venomous" else non_venomous_dest

        for img_file in item.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                dest_filename = f"{species_name}_{img_file.name}"
                dest_path = target_dir / dest_filename

                if mode == "symlink":
                    try:
                        if dest_path.exists():
                            dest_path.unlink()
                        dest_path.symlink_to(img_file.resolve())
                    except OSError:
                        # Graceful fallback to copy if OS privileges forbid symlinks (common on Windows)
                        shutil.copy2(img_file, dest_path)
                else:
                    shutil.copy2(img_file, dest_path)

                counts[target_class] += 1

    print("=" * 70)
    print("SPECIES TO BINARY DATASET COMPILATION COMPLETE")
    print(f"Source: {src_dir.resolve()}")
    print(f"Destination: {dest_dir.resolve()}")
    print(f"Transfer mode: {mode}")
    print(f"Images transferred - venomous: {counts['venomous']}, non_venomous: {counts['non_venomous']}")

    if unmapped_species:
        print(f"\n[NOTICE] {len(unmapped_species)} species folder(s) were not in '{map_file.name}' and were skipped:")
        for s in sorted(unmapped_species):
            print(f"  - {s}")
        print("Add them to configs/species_map.yaml if you want them included.")
    print("=" * 70)

    return counts


def main() -> None:
    """Parse CLI arguments and run species dataset restructuring."""
    parser = argparse.ArgumentParser(
        description="Reorganize species folder dataset into binary venomous/non_venomous folders."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to source directory containing species folders",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="data/raw",
        help="Path to destination directory (default: data/raw)",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default="configs/species_map.yaml",
        help="Path to species mapping YAML (default: configs/species_map.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help="Transfer mode ('copy' or 'symlink', default: copy)",
    )
    args = parser.parse_args()

    build_binary_from_species(
        src_dir=Path(args.src_dir),
        dest_dir=Path(args.dest_dir),
        map_file=Path(args.map_file),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
