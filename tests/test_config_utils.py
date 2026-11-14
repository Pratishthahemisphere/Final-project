"""Unit tests for configuration loading and utility functions."""

from pathlib import Path
import pytest
import torch

from src.config import Config, DataConfig, load_config
from src.utils import ensure_dirs, get_device, set_seed, setup_logger


def test_load_config_valid(tmp_path: Path) -> None:
    """Test loading a valid YAML configuration."""
    cfg_file = tmp_path / "config.yaml"
    cfg_content = """
seed: 123
paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  checkpoint_dir: "outputs/checkpoints"
  plots_dir: "outputs/plots"
  logs_dir: "outputs/logs"
data:
  img_size: 224
  batch_size: 16
  num_workers: 2
  val_split: 0.2
  test_split: 0.1
model:
  backbone: "resnet50"
  pretrained: false
  attention: "cbam"
  dropout: 0.25
  num_classes: 2
train:
  epochs: 10
  lr: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.1
  patience: 5
  amp: false
  freeze_backbone_epochs: 1
inference:
  uncertainty_threshold: 0.75
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")
    cfg = load_config(cfg_file)

    assert isinstance(cfg, Config)
    assert cfg.seed == 123
    assert cfg.data.batch_size == 16
    assert cfg.data.num_workers == 2
    assert cfg.data.val_split == 0.2
    assert isinstance(cfg.paths.raw_dir, Path)
    assert cfg.model.backbone == "resnet50"
    assert cfg.train.amp is False
    assert cfg.inference.uncertainty_threshold == 0.75


def test_num_workers_auto_detection() -> None:
    """Test num_workers auto-defaults to 0 on Windows and 4 otherwise when None."""
    data_cfg = DataConfig(num_workers=None)
    import sys
    expected = 0 if sys.platform.startswith("win") else 4
    assert data_cfg.num_workers == expected


def test_config_invalid_splits() -> None:
    """Test that invalid split values raise ValueError."""
    with pytest.raises(ValueError):
        DataConfig(val_split=0.6, test_split=0.5)


def test_get_device() -> None:
    """Test device auto-detection returns a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_set_seed() -> None:
    """Test reproducibility after set_seed."""
    set_seed(42)
    t1 = torch.rand(5)
    set_seed(42)
    t2 = torch.rand(5)
    assert torch.equal(t1, t2)


def test_setup_logger(tmp_path: Path) -> None:
    """Test logger initialization and file writing."""
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_logger", log_file=log_file)
    logger.info("Test message")
    assert log_file.exists()
    assert "Test message" in log_file.read_text(encoding="utf-8")


def test_ensure_dirs(tmp_path: Path) -> None:
    """Test ensure_dirs directory creation."""
    dir1 = tmp_path / "sub1" / "nested"
    dir2 = tmp_path / "sub2"
    ensure_dirs(dir1, dir2)
    assert dir1.is_dir()
    assert dir2.is_dir()
