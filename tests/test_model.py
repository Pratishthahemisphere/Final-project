"""Unit tests for CBAM modules, SnakeAttentionNet architectures, and build_model factory."""

import pytest
import torch

from src.config import ModelConfig
from src.model import (
    CBAM,
    ChannelAttention,
    SnakeAttentionNet,
    SpatialAttention,
    build_model,
)


def test_channel_attention_preserves_shape() -> None:
    """Verify ChannelAttention preserves spatial and channel dimensions."""
    ca = ChannelAttention(in_channels=64, reduction_ratio=16)
    x = torch.randn(2, 64, 28, 28)
    out = ca(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_spatial_attention_preserves_shape() -> None:
    """Verify SpatialAttention preserves spatial and channel dimensions."""
    sa = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 28, 28)
    out = sa(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_cbam_preserves_shape() -> None:
    """Verify combined CBAM module preserves tensor dimensions."""
    cbam = CBAM(in_channels=128, reduction_ratio=16, kernel_size=7)
    x = torch.randn(2, 128, 14, 14)
    out = cbam(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_model_output_shape_batch_4() -> None:
    """Verify SnakeAttentionNet produces output logits of shape [4, 2] for batch size 4."""
    model = SnakeAttentionNet(
        backbone="resnet18",  # lightweight for quick testing
        pretrained=False,
        attention="cbam",
        dropout=0.3,
        num_classes=2,
    )
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_freeze_unfreeze_parameter_counts() -> None:
    """Verify that freeze_backbone and unfreeze_backbone properly toggle requires_grad."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )

    counts_init = model.count_parameters()
    assert counts_init["trainable"] == counts_init["total"]

    # Freeze backbone
    model.freeze_backbone()
    counts_frozen = model.count_parameters()
    assert counts_frozen["trainable"] < counts_frozen["total"]
    assert counts_frozen["trainable"] > 0

    # Verify backbone parameters are non-trainable
    assert model.conv1.weight.requires_grad is False
    assert model.layer1[0].conv1.weight.requires_grad is False
    assert model.layer4[0].conv1.weight.requires_grad is False

    # Verify CBAM modules and head remain trainable
    assert model.cbam1.channel_attention.mlp[0].weight.requires_grad is True
    assert model.cbam4.spatial_attention.conv.weight.requires_grad is True
    assert model.fc.weight.requires_grad is True

    # Unfreeze backbone
    model.unfreeze_backbone()
    counts_unfrozen = model.count_parameters()
    assert counts_unfrozen["trainable"] == counts_unfrozen["total"]
    assert model.conv1.weight.requires_grad is True


def test_backward_pass_finite_gradients() -> None:
    """Verify that a full backward pass computes valid, finite gradients across trainable layers."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Check finite gradients on classification head and CBAM
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()

    assert model.cbam4.spatial_attention.conv.weight.grad is not None
    assert torch.isfinite(model.cbam4.spatial_attention.conv.weight.grad).all()


def test_eval_mode_batch_size_1() -> None:
    """Verify model functions correctly in eval mode with a single sample batch [1, 3, 224, 224]."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_target_layer_attribute() -> None:
    """Verify target_layer attribute is exposed for Grad-CAM hooking."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    assert hasattr(model, "target_layer")
    assert model.target_layer is model.cbam4
    assert model.get_target_layer() is model.cbam4


def test_build_model_factory_resnet50() -> None:
    """Verify build_model factory correctly constructs ResNet50 architecture."""
    cfg = ModelConfig(
        backbone="resnet50",
        pretrained=False,
        attention="cbam",
        dropout=0.3,
        num_classes=2,
    )
    model = build_model(cfg)
    assert isinstance(model, SnakeAttentionNet)
    assert model.backbone_name == "resnet50"
    assert model.fc.out_features == 2

    # ResNet50 layer4 output channel dimension is 2048
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
