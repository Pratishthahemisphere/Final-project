"""Dataset definitions, stratified data splitting, image transforms, and DataLoader factories."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config

logger = logging.getLogger(__name__)

# Constants for fixed label mapping: non_venomous=0, venomous=1
LABEL_NON_VENOMOUS: int = 0
LABEL_VENOMOUS: int = 1

CLASS_NAMES: list[str] = ["non_venomous", "venomous"]

CLASS_TO_IDX: dict[str, int] = {
    "non_venomous": LABEL_NON_VENOMOUS,
    "venomous": LABEL_VENOMOUS,
}

IDX_TO_CLASS: dict[int, str] = {
    LABEL_NON_VENOMOUS: "non_venomous",
    LABEL_VENOMOUS: "venomous",
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 checksum of a file in streaming chunks.

    Args:
        filepath: Path to file.

    Returns:
        str: MD5 hex digest.
    """
    hasher = hashlib.md5()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(filepath: Path) -> bool:
    """Check if image file is non-corrupted and meets minimum resolution (>=32x32).

    Args:
        filepath: Path to image file.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            return w >= 32 and h >= 32
    except Exception:  # noqa: BLE001
        return False


class SnakeDataset(Dataset):
    """PyTorch Dataset for snake image classification with automatic RGB conversion and error fallback."""

    def __init__(
        self,
        samples: list[tuple[str | Path, int]],
        transform: Any | None = None,
    ) -> None:
        """Initialize SnakeDataset.

        Args:
            samples: List of (image_path, integer_label) pairs.
            transform: Optional torchvision transform pipeline.
        """
        self.samples: list[tuple[Path, int]] = [
            (Path(p), int(label)) for p, label in samples
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return image tensor and label.

        If an image fails to load at runtime, logs a warning and returns
        the next valid sample rather than crashing.
        """
        max_attempts = len(self.samples)
        current_idx = idx

        for _ in range(max_attempts):
            img_path, label = self.samples[current_idx]
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                if self.transform is not None:
                    img_tensor = self.transform(img_rgb)
                else:
                    img_tensor = T.functional.to_tensor(img_rgb)
                return img_tensor, label
            except Exception as e:  # noqa: BLE001  # noqa: BLE001
                logger.warning(
                    f"Failed to load image at '{img_path}': {e}. Returning next sample."
                )
                current_idx = (current_idx + 1) % len(self.samples)

        raise RuntimeError(f"All {max_attempts} images in SnakeDataset failed to load.")


def get_train_transform(img_size: int = 224) -> T.Compose:
    """Return training image augmentation pipeline.

    Applies RandomResizedCrop, RandomHorizontalFlip, RandomRotation(15),
    and ColorJitter with mild hue shifts to protect snake coloration features.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(img_size: int = 224) -> T.Compose:
    """Return deterministic validation/test image transformation pipeline."""
    resize_dim = int(img_size * 256 / 224)
    return T.Compose(
        [
            T.Resize(resize_dim),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_splits(
    raw_dir: str | Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    processed_dir: str | Path = "data/processed",
    force_rebuild: bool = False,
) -> dict[str, list[tuple[Path, int]]]:
    """Scan raw image directory, deduplicate by MD5, and construct stratified train/val/test splits.

    Caches the split metadata to data/processed/splits.json and reloads it on subsequent
    runs if the dataset content hash matches.

    Args:
        raw_dir: Path to directory containing 'venomous' and 'non_venomous' subfolders.
        val_split: Fraction of dataset allocated to validation (e.g. 0.15).
        test_split: Fraction of dataset allocated to testing (e.g. 0.15).
        seed: Random seed for deterministic splitting.
        processed_dir: Directory where splits.json will be saved/loaded.
        force_rebuild: If True, ignores cached splits and recomputes.

    Returns:
        Dict[str, List[Tuple[Path, int]]]: Mapping with keys 'train', 'val', 'test',
        each containing lists of (Path, integer_label) pairs.

    Raises:
        FileNotFoundError: If raw_dir or required class folders do not exist.
        ValueError: If splits are invalid or any split contains fewer than 2 images per class.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    splits_file = processed_path / "splits.json"

    venomous_dir = raw_path / "venomous"
    non_venomous_dir = raw_path / "non_venomous"

    if not raw_path.is_dir():
        raise FileNotFoundError(
            f"Dataset raw directory '{raw_path.resolve()}' not found."
        )
    if not venomous_dir.is_dir() or not non_venomous_dir.is_dir():
        raise FileNotFoundError(
            f"Both 'venomous' and 'non_venomous' directories must exist inside '{raw_path.resolve()}'."
        )

    # 1. Gather all files, skip corrupted/tiny, and deduplicate by MD5 before splitting
    seen_md5: dict[str, tuple[str, Path]] = {}
    valid_samples: list[tuple[Path, int]] = []
    file_fingerprint_entries: list[str] = []

    for cls_name, cls_label in [("non_venomous", LABEL_NON_VENOMOUS), ("venomous", LABEL_VENOMOUS)]:
        cls_dir = raw_path / cls_name
        for f in sorted(cls_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue

            if not is_valid_image(f):
                logger.warning(f"Skipping corrupted or tiny image: {f}")
                continue

            file_md5 = compute_file_md5(f)
            if file_md5 in seen_md5:
                prev_cls, prev_path = seen_md5[file_md5]
                logger.warning(
                    f"Duplicate image skipped: '{f}' (class {cls_name}) is identical to '{prev_path}' (class {prev_cls})."
                )
                continue

            seen_md5[file_md5] = (cls_name, f)
            valid_samples.append((f, cls_label))
            file_fingerprint_entries.append(f"{f.name}:{file_md5}")

    # Compute dataset fingerprint
    dataset_hash = hashlib.sha256(
        ",".join(sorted(file_fingerprint_entries)).encode("utf-8")
    ).hexdigest()

    # 2. Check if cached splits match current dataset and parameters
    if not force_rebuild and splits_file.is_file():
        try:
            with splits_file.open("r", encoding="utf-8") as sf:
                cached = json.load(sf)
            if (
                cached.get("dataset_hash") == dataset_hash
                and cached.get("val_split") == val_split
                and cached.get("test_split") == test_split
                and cached.get("seed") == seed
            ):
                logger.info(f"Loaded existing valid splits from '{splits_file}'.")
                return {
                    split_name: [
                        (Path(p), int(label))
                        for p, label in cached[split_name]
                    ]
                    for split_name in ["train", "val", "test"]
                }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read existing splits.json ({e}). Rebuilding splits.")

    # 3. Validate sample counts
    c0 = sum(1 for _, label in valid_samples if label == LABEL_NON_VENOMOUS)
    c1 = sum(1 for _, label in valid_samples if label == LABEL_VENOMOUS)
    if c0 < 2 or c1 < 2:
        raise ValueError(
            f"Dataset in '{raw_path}' has insufficient samples for splitting. "
            f"Need >=2 samples per class, found non_venomous={c0}, venomous={c1}."
        )

    # 4. Perform stratified train/val/test split
    all_paths = [p for p, _ in valid_samples]
    all_labels = [label for _, label in valid_samples]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=test_split,
        random_state=seed,
        stratify=all_labels,
    )

    val_relative_size = val_split / (1.0 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_relative_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    splits: dict[str, list[tuple[Path, int]]] = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }

    # 5. Verify that each split contains at least 2 images for every class
    for split_name, samples in splits.items():
        count_0 = sum(1 for _, y in samples if y == LABEL_NON_VENOMOUS)
        count_1 = sum(1 for _, y in samples if y == LABEL_VENOMOUS)
        if count_0 < 2 or count_1 < 2:
            raise ValueError(
                f"Split '{split_name}' contains fewer than 2 images in a class: "
                f"non_venomous={count_0}, venomous={count_1}. "
                "Adjust splits or supply more data."
            )

    # 6. Save splits to data/processed/splits.json
    splits_payload = {
        "dataset_hash": dataset_hash,
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "raw_dir": str(raw_path),
        "train": [(str(p.as_posix()), int(lbl)) for p, lbl in splits["train"]],
        "val": [(str(p.as_posix()), int(lbl)) for p, lbl in splits["val"]],
        "test": [(str(p.as_posix()), int(lbl)) for p, lbl in splits["test"]],
    }

    with splits_file.open("w", encoding="utf-8") as sf:
        json.dump(splits_payload, sf, indent=2)

    logger.info(
        f"Saved deterministic dataset splits to '{splits_file}' "
        f"(train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])})."
    )

    return splits


def get_dataloaders(
    cfg: Config,
    raw_dir: str | Path | None = None,
    force_rebuild_splits: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor | None]:
    """Construct PyTorch DataLoaders and compute class weights from the TRAIN split only.

    Supports either a WeightedRandomSampler OR class-weighted loss based on config,
    preventing double-weighting.

    Args:
        cfg: Configuration dataclass.
        raw_dir: Optional override for raw dataset directory (e.g. for smoke tests).
        force_rebuild_splits: Whether to force re-splitting rather than using cached splits.

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - test_loader: DataLoader for test set
            - class_weights: torch.Tensor of shape (2,) for class-weighted loss,
              or None if WeightedRandomSampler is used.
    """
    effective_raw_dir = Path(raw_dir) if raw_dir is not None else cfg.paths.raw_dir

    splits = build_splits(
        raw_dir=effective_raw_dir,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
        processed_dir=cfg.paths.processed_dir,
        force_rebuild=force_rebuild_splits,
    )

    train_transform = get_train_transform(cfg.data.img_size)
    val_transform = get_val_transform(cfg.data.img_size)

    train_dataset = SnakeDataset(splits["train"], transform=train_transform)
    val_dataset = SnakeDataset(splits["val"], transform=val_transform)
    test_dataset = SnakeDataset(splits["test"], transform=val_transform)

    # Compute class weights strictly on the TRAIN split
    train_labels = [label for _, label in splits["train"]]
    count_0 = sum(1 for y in train_labels if y == LABEL_NON_VENOMOUS)
    count_1 = sum(1 for y in train_labels if y == LABEL_VENOMOUS)
    total_train = len(train_labels)

    w0 = total_train / (2.0 * count_0) if count_0 > 0 else 1.0
    w1 = total_train / (2.0 * count_1) if count_1 > 0 else 1.0
    computed_class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    sampler_choice = getattr(cfg.data, "sampler", "none").lower()

    pin_memory = torch.cuda.is_available()
    persistent_workers = cfg.data.num_workers > 0

    if sampler_choice == "weighted":
        # Use WeightedRandomSampler; disable class_weights in loss to prevent double-weighting
        sample_weights = [
            w1 if y == LABEL_VENOMOUS else w0 for y in train_labels
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        loss_class_weights: torch.Tensor | None = None
    else:
        # Standard shuffle; return class_weights for use in CrossEntropyLoss
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        loss_class_weights = computed_class_weights

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, loss_class_weights
