"""Unit tests for dataset loading, stratification, splits caching, and transforms."""

from pathlib import Path
from PIL import Image
import pytest
import torch

from src.config import Config, DataConfig, InferenceConfig, ModelConfig, PathConfig, TrainConfig
from src.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    LABEL_NON_VENOMOUS,
    LABEL_VENOMOUS,
    SnakeDataset,
    build_splits,
    get_dataloaders,
    get_train_transform,
    get_val_transform,
)


def create_dummy_dataset(base_dir: Path, n_per_class: int = 20, img_size: int = 64) -> Path:
    """Helper to generate a clean synthetic dataset for testing."""
    for cls_idx, cls_name in enumerate(["non_venomous", "venomous"]):
        cls_dir = base_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            # Ensure unique pixel values across both classes and image indices
            r = (cls_idx * 128 + i * 5) % 256
            g = ((1 - cls_idx) * 128 + i * 7) % 256
            b = (i * 11 + 40) % 256
            img = Image.new("RGB", (img_size, img_size), color=(r, g, b))
            img.save(cls_dir / f"img_{i:03d}.jpg")
    return base_dir


def test_label_constants_and_mapping() -> None:
    """Verify that class-to-index mappings adhere strictly to project specification."""
    assert LABEL_NON_VENOMOUS == 0
    assert LABEL_VENOMOUS == 1
    assert CLASS_TO_IDX["non_venomous"] == 0
    assert CLASS_TO_IDX["venomous"] == 1
    assert IDX_TO_CLASS[0] == "non_venomous"
    assert IDX_TO_CLASS[1] == "venomous"


def test_splits_disjoint_and_stratification(tmp_path: Path) -> None:
    """Verify train, val, and test splits are strictly disjoint and maintain class stratification."""
    raw_dir = create_dummy_dataset(tmp_path / "raw", n_per_class=20)
    proc_dir = tmp_path / "processed"

    splits = build_splits(
        raw_dir=raw_dir,
        val_split=0.20,
        test_split=0.20,
        seed=42,
        processed_dir=proc_dir,
    )

    train_paths = {p.resolve() for p, _ in splits["train"]}
    val_paths = {p.resolve() for p, _ in splits["val"]}
    test_paths = {p.resolve() for p, _ in splits["test"]}

    # Verify no data leakage between splits (disjoint sets)
    assert train_paths.isdisjoint(val_paths), "Train and Val splits share images!"
    assert train_paths.isdisjoint(test_paths), "Train and Test splits share images!"
    assert val_paths.isdisjoint(test_paths), "Val and Test splits share images!"

    # Verify total count matches input (40 unique images)
    total_split_count = len(train_paths) + len(val_paths) + len(test_paths)
    assert total_split_count == 40

    # Verify stratification (each split should have 50% non-venomous, 50% venomous)
    for split_name in ["train", "val", "test"]:
        samples = splits[split_name]
        c0 = sum(1 for _, y in samples if y == LABEL_NON_VENOMOUS)
        c1 = sum(1 for _, y in samples if y == LABEL_VENOMOUS)
        assert c0 == c1, f"Stratification failed for split '{split_name}': {c0} vs {c1}"
        assert c0 >= 2, f"Expected at least 2 images per class in '{split_name}', got {c0}"


def test_corrupted_and_tiny_files_skipped(tmp_path: Path) -> None:
    """Verify corrupted and undersized files are skipped during split construction."""
    raw_dir = create_dummy_dataset(tmp_path / "raw", n_per_class=10)

    # Add corrupted file
    corrupt_file = raw_dir / "venomous" / "corrupted.jpg"
    corrupt_file.write_bytes(b"This is not a real JPEG image file.")

    # Add tiny file (<32px)
    tiny_img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    tiny_img.save(raw_dir / "non_venomous" / "tiny.jpg")

    splits = build_splits(
        raw_dir=raw_dir,
        val_split=0.2,
        test_split=0.2,
        seed=42,
        processed_dir=tmp_path / "processed",
        force_rebuild=True,
    )

    all_paths = [p.resolve() for split in splits.values() for p, _ in split]
    assert corrupt_file.resolve() not in all_paths
    assert (raw_dir / "non_venomous" / "tiny.jpg").resolve() not in all_paths


def test_dataset_error_fallback(tmp_path: Path) -> None:
    """Verify that SnakeDataset.__getitem__ falls back gracefully to the next sample on read error."""
    # Create one valid image and one corrupted image
    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (64, 64), color=(0, 200, 0)).save(valid_path)

    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"corrupted header bytes")

    dataset = SnakeDataset(
        samples=[(corrupt_path, LABEL_NON_VENOMOUS), (valid_path, LABEL_VENOMOUS)],
        transform=get_val_transform(img_size=224),
    )

    # Request index 0 (which is corrupted) - should return sample 1 instead of crashing
    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert label == LABEL_VENOMOUS


def test_dataloaders_batch_shape_and_types(tmp_path: Path) -> None:
    """Verify DataLoader outputs have shape [B, 3, 224, 224] and appropriate label types."""
    raw_dir = create_dummy_dataset(tmp_path / "raw", n_per_class=15)
    proc_dir = tmp_path / "processed"

    cfg = Config(
        seed=42,
        paths=PathConfig(
            raw_dir=raw_dir,
            processed_dir=proc_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            plots_dir=tmp_path / "plots",
            logs_dir=tmp_path / "logs",
        ),
        data=DataConfig(
            img_size=224,
            batch_size=8,
            num_workers=0,
            val_split=0.2,
            test_split=0.2,
            sampler="none",
        ),
        model=ModelConfig(),
        train=TrainConfig(),
        inference=InferenceConfig(),
    )

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(cfg)

    # Check batch shapes
    images, labels = next(iter(train_loader))
    assert images.ndim == 4
    assert images.shape == (8, 3, 224, 224)
    assert labels.ndim == 1
    assert labels.shape == (8,)
    assert labels.dtype == torch.int64

    # Check class weights tensor
    assert class_weights is not None
    assert class_weights.shape == (2,)
    assert class_weights.dtype == torch.float32


def test_splits_caching(tmp_path: Path) -> None:
    """Verify splits are cached to splits.json and reloaded without changes."""
    raw_dir = create_dummy_dataset(tmp_path / "raw", n_per_class=20)
    proc_dir = tmp_path / "processed"

    splits_first = build_splits(raw_dir=raw_dir, processed_dir=proc_dir, seed=42)
    splits_file = proc_dir / "splits.json"
    assert splits_file.is_file()

    # Second call should load from cache
    splits_second = build_splits(raw_dir=raw_dir, processed_dir=proc_dir, seed=42)
    assert splits_first["train"] == splits_second["train"]
    assert splits_first["val"] == splits_second["val"]
    assert splits_first["test"] == splits_second["test"]
