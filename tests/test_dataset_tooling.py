"""Tests for dataset tooling scripts: make_smoke_data, check_dataset, and build_binary_from_species."""

from pathlib import Path
from PIL import Image
import pytest

from scripts.build_binary_from_species import build_binary_from_species, load_species_mapping
from scripts.check_dataset import check_dataset, compute_md5, verify_image
from scripts.make_smoke_data import make_smoke_dataset


def test_make_smoke_dataset(tmp_path: Path) -> None:
    """Test generating smoke dataset creates expected number and size of images."""
    out_dir = tmp_path / "smoke_test"
    make_smoke_dataset(out_dir=out_dir, n_per_class=5, img_size=64)

    venomous_files = list((out_dir / "venomous").glob("*.jpg"))
    non_venomous_files = list((out_dir / "non_venomous").glob("*.jpg"))

    assert len(venomous_files) == 5
    assert len(non_venomous_files) == 5

    # Check image properties
    with Image.open(venomous_files[0]) as img:
        assert img.size == (64, 64)
        assert img.mode == "RGB"


def test_check_dataset_valid(tmp_path: Path) -> None:
    """Test check_dataset succeeds on clean generated data."""
    out_dir = tmp_path / "valid_data"
    make_smoke_dataset(out_dir=out_dir, n_per_class=10, img_size=64)
    result = check_dataset(out_dir)
    assert result is True


def test_check_dataset_missing_folder(tmp_path: Path) -> None:
    """Test check_dataset fails when class folders are missing."""
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    result = check_dataset(empty_dir)
    assert result is False


def test_check_dataset_cross_class_duplicate(tmp_path: Path) -> None:
    """Test that exact cross-class duplicates trigger fatal failure."""
    data_dir = tmp_path / "dup_data"
    v_dir = data_dir / "venomous"
    nv_dir = data_dir / "non_venomous"
    v_dir.mkdir(parents=True)
    nv_dir.mkdir(parents=True)

    # Create identical image in both classes
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(v_dir / "snake.jpg")
    img.save(nv_dir / "snake.jpg")

    result = check_dataset(data_dir)
    assert result is False


def test_build_binary_from_species(tmp_path: Path) -> None:
    """Test species mapping and restructuring."""
    species_root = tmp_path / "species_src"
    naja_dir = species_root / "Naja_naja"
    corn_dir = species_root / "Pantherophis_guttatus"
    naja_dir.mkdir(parents=True)
    corn_dir.mkdir(parents=True)

    # Dummy images
    img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    img.save(naja_dir / "cobra_1.png")
    img.save(corn_dir / "corn_1.jpg")

    map_file = tmp_path / "map.yaml"
    map_file.write_text(
        """
species_mapping:
  "Naja_naja": "venomous"
  "Pantherophis_guttatus": "non_venomous"
""",
        encoding="utf-8",
    )

    dest_dir = tmp_path / "binary_dest"
    counts = build_binary_from_species(species_root, dest_dir, map_file, mode="copy")

    assert counts["venomous"] == 1
    assert counts["non_venomous"] == 1
    assert (dest_dir / "venomous" / "Naja_naja_cobra_1.png").exists()
    assert (dest_dir / "non_venomous" / "Pantherophis_guttatus_corn_1.jpg").exists()
