"""Custom Grad-CAM and CBAM attention interpretability module from scratch with PyTorch hooks.

Usage:
    python -m src.explain --image PATH --checkpoint PATH [--out outputs/plots/explain.png]
"""

import argparse
import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.config import load_config
from src.dataset import IDX_TO_CLASS, get_val_transform
from src.model import SnakeAttentionNet
from src.utils import get_device

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) engine.

    Registers PyTorch forward and full backward hooks on target convolutional layer,
    computes class activation maps, upsamples to input resolution, safely normalizes
    to [0, 1], and properly removes hooks upon completion.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize GradCAM.

        Args:
            model: PyTorch classification model.
            target_layer: Target convolutional module to hook into.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hook_handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module, grad_inp: Any, grad_out: tuple[torch.Tensor, ...]
        ) -> None:
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Deregister all active hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute normalized Grad-CAM heatmap for target input tensor.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 3, H, W).
            target_class: Class index to explain (default: argmax prediction).

        Returns:
            Tuple[np.ndarray, int, float]:
                - heatmap: 2D numpy array of shape (H, W) in [0, 1]
                - predicted_class: integer index of predicted class
                - confidence: softmax probability of predicted class
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            tensor = input_tensor.clone().requires_grad_(True)
            self.model.zero_grad()
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)

            pred_class = int(torch.argmax(outputs, dim=1).item())
            confidence = float(probs[0, pred_class].item())

            chosen_class = target_class if target_class is not None else pred_class

            # Backward pass on target class logit score
            score = outputs[0, chosen_class]
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed: Hooks did not capture activations or gradients.")

        # Compute importance weights alpha_k via Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Positive influences only

        # Upsample to match input spatial dimensions
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam_upsampled = F.interpolate(
            cam, size=(h_in, w_in), mode="bilinear", align_corners=False
        )
        cam_np = cam_upsampled.squeeze().cpu().numpy()

        # Safe normalization to [0, 1], guarding against division by zero
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if (c_max - c_min) > 1e-8:
            heatmap = (cam_np - c_min) / (c_max - c_min)
        else:
            heatmap = np.zeros_like(cam_np, dtype=np.float32)

        return heatmap, pred_class, confidence


def extract_cbam_spatial_attention(
    model: SnakeAttentionNet,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Extract and upsample the CBAM spatial attention heatmap from the final stage (cbam4).

    Args:
        model: SnakeAttentionNet in eval mode.
        input_tensor: Preprocessed tensor of shape (1, 3, H, W).

    Returns:
        np.ndarray: 2D numpy array of shape (H, W) in [0, 1].
    """
    model.eval()
    h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]

    if not model.use_attention or not hasattr(model, "cbam4") or not hasattr(model.cbam4, "spatial_attention"):
        return np.zeros((h_in, w_in), dtype=np.float32)

    spatial_weights_capture: list[torch.Tensor] = []

    def hook_spatial(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
        spatial_weights_capture.append(output.detach())

    # Hook the final sigmoid of the spatial attention module in cbam4
    hook_handle = model.cbam4.spatial_attention.sigmoid.register_forward_hook(hook_spatial)

    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        hook_handle.remove()

    if not spatial_weights_capture:
        return np.zeros((h_in, w_in), dtype=np.float32)

    # Tensor shape is (1, 1, h_feat, w_feat)
    raw_map = spatial_weights_capture[0]
    upsampled = F.interpolate(
        raw_map, size=(h_in, w_in), mode="bilinear", align_corners=False
    )
    att_np = upsampled.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    a_min, a_max = float(att_np.min()), float(att_np.max())
    if (a_max - a_min) > 1e-8:
        norm_att = (att_np - a_min) / (a_max - a_min)
    else:
        norm_att = np.zeros_like(att_np, dtype=np.float32)

    return norm_att


def overlay_heatmap(
    original_pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Blend a jet colormap representation of the heatmap onto the original PIL image.

    Args:
        original_pil_image: Base PIL image.
        heatmap: 2D float numpy array in [0, 1].
        alpha: Heatmap blend opacity in [0, 1] (default: 0.4).

    Returns:
        PIL.Image: Blended RGB visualization image.
    """
    pil_rgb = original_pil_image.convert("RGB")
    w, h = pil_rgb.size

    # Resize heatmap to match image dimensions
    if heatmap.shape != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # Clamp and convert to uint8
    heatmap_clamped = np.clip(heatmap_resized, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap_clamped)

    # OpenCV applyColorMap produces BGR
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    orig_arr = np.array(pil_rgb, dtype=np.float32) / 255.0
    heat_arr = heatmap_rgb.astype(np.float32) / 255.0

    blended = (1.0 - alpha) * orig_arr + alpha * heat_arr
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)


def generate_explanation_figure(
    image_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    img_size: int = 224,
) -> None:
    """Generate and save 3-panel explanation figure (Original, Grad-CAM, CBAM Attention).

    Args:
        image_path: Path to target input snake image.
        checkpoint_path: Path to trained model checkpoint (.pt).
        out_path: Destination path for output PNG visualization.
        img_size: Model input dimension.
    """
    device = get_device()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image '{image_path.resolve()}' not found.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path.resolve()}' not found.")

    # 1. Load Checkpoint and Rebuild Model
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    attention = ckpt.get("attention", "cbam")

    model = SnakeAttentionNet(
        backbone=backbone,
        pretrained=False,
        attention=attention,
        dropout=0.3,
        num_classes=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Preprocess Image
    pil_img = Image.open(image_path).convert("RGB")
    val_transform = get_val_transform(img_size=img_size)
    tensor = val_transform(pil_img).unsqueeze(0).to(device)

    # 3. Compute Grad-CAM
    with GradCAM(model, model.get_target_layer()) as cam_engine:
        gradcam_map, pred_class, confidence = cam_engine.generate_heatmap(tensor)

    # 4. Extract CBAM Spatial Attention Map
    cbam_map = extract_cbam_spatial_attention(model, tensor)

    # 5. Overlays
    gradcam_overlay = overlay_heatmap(pil_img, gradcam_map, alpha=0.45)
    cbam_overlay = overlay_heatmap(pil_img, cbam_map, alpha=0.45)

    # 6. Plot 3 Panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(pil_img)
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM Class Activation", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cbam_overlay)
    axes[2].set_title("CBAM Spatial Attention", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    pred_name = IDX_TO_CLASS[pred_class]
    header_color = "darkred" if pred_name == "venomous" else "darkgreen"

    plt.suptitle(
        f"Model Prediction: {pred_name.upper()}  |  Confidence: {confidence * 100:.1f}%",
        fontsize=15,
        fontweight="bold",
        color=header_color,
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"SUCCESS: Saved 3-panel explanation visualization to '{out_path.resolve()}'.")


def main() -> None:
    """Parse CLI arguments and run explanation generator."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM and CBAM visual explanations for snake detection."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint (default: outputs/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/plots/explain.png",
        help="Output plot file path (default: outputs/plots/explain.png)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_explanation_figure(
        image_path=Path(args.image),
        checkpoint_path=Path(args.checkpoint),
        out_path=Path(args.out),
        img_size=cfg.data.img_size,
    )


if __name__ == "__main__":
    main()
