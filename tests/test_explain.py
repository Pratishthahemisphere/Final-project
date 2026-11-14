"""Unit tests for Grad-CAM, CBAM attention visualization, and heatmap blending."""

import numpy as np
from PIL import Image
import pytest
import torch

from src.explain import (
    GradCAM,
    extract_cbam_spatial_attention,
    overlay_heatmap,
)
from src.model import SnakeAttentionNet


def test_gradcam_heatmap_properties() -> None:
    """Verify Grad-CAM produces [0, 1] normalized heatmaps matching input dimensions."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    x = torch.randn(1, 3, 224, 224)

    with GradCAM(model, model.get_target_layer()) as cam_engine:
        heatmap, pred_class, confidence = cam_engine.generate_heatmap(x)

    assert heatmap.shape == (224, 224)
    assert not np.isnan(heatmap).any(), "Heatmap contains NaN values!"
    assert float(heatmap.min()) >= 0.0, "Heatmap minimum below 0.0"
    assert float(heatmap.max()) <= 1.0, "Heatmap maximum exceeds 1.0"
    assert pred_class in [0, 1]
    assert 0.0 <= confidence <= 1.0


def test_gradcam_hooks_removed_properly() -> None:
    """Verify all forward and backward hooks are removed when exiting context manager or calling remove_hooks."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )

    cam_engine = GradCAM(model, model.get_target_layer())
    assert len(cam_engine.hook_handles) == 2

    # Explicit removal
    cam_engine.remove_hooks()
    assert len(cam_engine.hook_handles) == 0

    # Context manager auto-removal
    with GradCAM(model, model.get_target_layer()) as ctx_engine:
        assert len(ctx_engine.hook_handles) == 2
    assert len(ctx_engine.hook_handles) == 0


def test_extract_cbam_spatial_attention() -> None:
    """Verify CBAM spatial attention map extraction, shape, and normalization."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    x = torch.randn(1, 3, 224, 224)

    att_map = extract_cbam_spatial_attention(model, x)
    assert att_map.shape == (224, 224)
    assert not np.isnan(att_map).any(), "Attention map contains NaN values!"
    assert float(att_map.min()) >= 0.0
    assert float(att_map.max()) <= 1.0


def test_overlay_heatmap_output() -> None:
    """Verify overlay_heatmap produces valid RGB PIL image with matching dimensions."""
    base_img = Image.new("RGB", (200, 150), color=(100, 150, 200))
    heatmap = np.linspace(0.0, 1.0, 200 * 150, dtype=np.float32).reshape(150, 200)

    overlay = overlay_heatmap(base_img, heatmap, alpha=0.4)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 150)
    assert overlay.mode == "RGB"


def test_gradcam_all_zero_safe_handling() -> None:
    """Verify that an all-zero gradient condition normalizes cleanly without division by zero."""
    model = SnakeAttentionNet(
        backbone="resnet18",
        pretrained=False,
        attention="cbam",
        num_classes=2,
    )
    x = torch.zeros(1, 3, 224, 224)

    with GradCAM(model, model.get_target_layer()) as cam_engine:
        heatmap, _, _ = cam_engine.generate_heatmap(x)

    assert heatmap.shape == (224, 224)
    assert not np.isnan(heatmap).any()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
