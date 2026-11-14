"""Unit tests for evaluate module and clinical safety metrics."""

import json
from pathlib import Path
from PIL import Image
import pytest
import torch

from src.config import Config, DataConfig, InferenceConfig, ModelConfig, PathConfig, TrainConfig
from src.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, LABEL_NON_VENOMOUS, LABEL_VENOMOUS
from src.evaluate import evaluate_model, resolve_sample_path
from src.model import SnakeAttentionNet


def test_resolve_sample_path(tmp_path: Path) -> None:
    """Verify sample path resolution for existing and overridden paths."""
    real_file = tmp_path / "img1.jpg"
    real_file.write_bytes(b"dummy")

    # Absolute / direct
    assert resolve_sample_path(str(real_file)) == real_file

    # Overridden data directory
    nested = tmp_path / "venomous" / "snake.jpg"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"dummy")

    resolved = resolve_sample_path("data/raw/venomous/snake.jpg", override_data_dir=tmp_path)
    assert resolved == nested


def test_evaluate_model_execution(tmp_path: Path) -> None:
    """Test full evaluation pipeline on synthetic test set and dummy checkpoint."""
    # Create test images
    raw_dir = tmp_path / "raw"
    for cls in ["non_venomous", "venomous"]:
        d = raw_dir / cls
        d.mkdir(parents=True)
        for i in range(4):
            img = Image.new("RGB", (64, 64), color=(i * 20, 100, 50))
            img.save(d / f"img_{i}.jpg")

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    splits_file = proc_dir / "splits.json"

    splits_data = {
        "dataset_hash": "test_hash",
        "seed": 42,
        "val_split": 0.2,
        "test_split": 0.2,
        "train": [],
        "val": [],
        "test": [
            [str((raw_dir / "non_venomous" / f"img_{i}.jpg").as_posix()), 0] for i in range(4)
        ] + [
            [str((raw_dir / "venomous" / f"img_{i}.jpg").as_posix()), 1] for i in range(4)
        ],
    }
    with splits_file.open("w", encoding="utf-8") as f:
        json.dump(splits_data, f)

    # Create dummy trained checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best.pt"

    model = SnakeAttentionNet(backbone="resnet18", pretrained=False, attention="cbam", num_classes=2)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "monitored_metric": "val_venomous_f1",
        "best_score": 0.85,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "class_names": CLASS_NAMES,
        "img_size": 64,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "backbone": "resnet18",
        "attention": "cbam",
    }
    torch.save(payload, ckpt_path)

    cfg = Config(
        seed=42,
        paths=PathConfig(
            raw_dir=raw_dir,
            processed_dir=proc_dir,
            checkpoint_dir=ckpt_dir,
            plots_dir=tmp_path / "plots",
            logs_dir=tmp_path / "logs",
        ),
        data=DataConfig(img_size=64, batch_size=4),
        model=ModelConfig(backbone="resnet18", pretrained=False, attention="cbam"),
        train=TrainConfig(),
        inference=InferenceConfig(),
    )

    metrics = evaluate_model(cfg, checkpoint_path=str(ckpt_path))

    assert "accuracy" in metrics
    assert "venomous_recall" in metrics
    assert "false_safe_rate" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["venomous_recall"] + metrics["false_safe_rate"] == pytest.approx(1.0)

    # Verify generated plots
    assert (tmp_path / "plots" / "confusion_matrix.png").is_file()
    assert (tmp_path / "plots" / "roc_curve.png").is_file()
    assert (tmp_path / "plots" / "pr_curve.png").is_file()
    assert (tmp_path / "plots" / "misclassified_examples.png").is_file()
    assert (tmp_path / "logs" / "test_metrics.json").is_file()
    assert (tmp_path / "logs" / "threshold_analysis.csv").is_file()
