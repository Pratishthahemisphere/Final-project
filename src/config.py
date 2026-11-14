"""Configuration dataclasses and loader with schema validation."""

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathConfig:
    """Paths used throughout the project."""
    raw_dir: Path
    processed_dir: Path
    checkpoint_dir: Path
    plots_dir: Path
    logs_dir: Path

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.plots_dir = Path(self.plots_dir)
        self.logs_dir = Path(self.logs_dir)


@dataclass
class DataConfig:
    """Dataset and dataloader configurations."""
    img_size: int = 224
    batch_size: int = 32
    num_workers: int | None = None
    val_split: float = 0.15
    test_split: float = 0.15
    sampler: str = "none"

    def __post_init__(self) -> None:
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if not (0.0 < self.test_split < 1.0):
            raise ValueError(f"test_split must be in (0, 1), got {self.test_split}")
        if self.val_split + self.test_split >= 1.0:
            raise ValueError(
                f"Sum of val_split ({self.val_split}) and test_split ({self.test_split}) "
                f"must be strictly less than 1.0"
            )
        if self.sampler not in {"none", "weighted"}:
            raise ValueError(f"sampler must be 'none' or 'weighted', got '{self.sampler}'")

        # Rule 5: Set num_workers default to 0 on Windows and 4 otherwise, determined automatically in code.
        if self.num_workers is None or self.num_workers < 0:
            is_windows = sys.platform.startswith("win") or platform.system() == "Windows"
            self.num_workers = 0 if is_windows else 4


@dataclass
class ModelConfig:
    """Neural network architecture hyperparameters."""
    backbone: str = "resnet50"
    pretrained: bool = True
    attention: str = "cbam"
    dropout: float = 0.3
    num_classes: int = 2

    def __post_init__(self) -> None:
        if not self.backbone:
            raise ValueError("backbone must be a non-empty string")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")


@dataclass
class TrainConfig:
    """Training loop hyperparameters and optimization settings."""
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 7
    amp: bool = True
    freeze_backbone_epochs: int = 3

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.freeze_backbone_epochs < 0:
            raise ValueError(
                f"freeze_backbone_epochs must be non-negative, got {self.freeze_backbone_epochs}"
            )


@dataclass
class InferenceConfig:
    """Inference settings and thresholds."""
    uncertainty_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not (0.0 <= self.uncertainty_threshold <= 1.0):
            raise ValueError(
                f"uncertainty_threshold must be in [0, 1], got {self.uncertainty_threshold}"
            )


@dataclass
class Config:
    """Master configuration structure combining all sections."""
    seed: int
    paths: PathConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig


def load_config(config_path: str | Path = "configs/config.yaml") -> Config:
    """Load, parse, and validate the project YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config: Populated and validated Config dataclass instance.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If any configuration parameters fail validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at '{path.resolve()}'. "
            "Please ensure configs/config.yaml exists."
        )

    with path.open("r", encoding="utf-8") as f:
        raw_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    try:
        seed = int(raw_cfg.get("seed", 42))
        paths = PathConfig(**raw_cfg.get("paths", {}))
        data = DataConfig(**raw_cfg.get("data", {}))
        model = ModelConfig(**raw_cfg.get("model", {}))
        train = TrainConfig(**raw_cfg.get("train", {}))
        inference = InferenceConfig(**raw_cfg.get("inference", {}))
    except (TypeError, KeyError) as e:
        raise ValueError(f"Configuration file schema error in '{path}': {e}") from e

    return Config(
        seed=seed,
        paths=paths,
        data=data,
        model=model,
        train=train,
        inference=inference,
    )
