"""Attention-augmented Convolutional Neural Network architectures for snake detection.

Implements the Convolutional Block Attention Module (CBAM) and integrates it across
the residual stages of deep convolutional backbones (ResNet50 / ResNet18).
"""

import logging

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from src.config import Config, ModelConfig

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM.

    Applies a shared two-layer MLP over both Average-Pooled and Max-Pooled feature maps,
    summing their outputs and scaling through a Sigmoid activation.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        """Initialize ChannelAttention.

        Args:
            in_channels: Number of input channels.
            reduction_ratio: Bottleneck reduction ratio for the shared MLP (default: 16).
        """
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented via 1x1 convolutions for spatial invariance and tensor efficiency
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of the shared MLP."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Channel-scaled feature tensor of shape (B, C, H, W).
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """Spatial Attention Module of CBAM.

    Computes channel-wise average and max projections, concatenates them,
    and applies a single bias-free 7x7 convolution followed by Sigmoid.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize SpatialAttention.

        Args:
            kernel_size: Convolutional kernel size (default: 7). Must be odd.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"SpatialAttention kernel_size must be odd, got {kernel_size}")

        padding = kernel_size // 2
        # Input has 2 channels: 1 from channel-wise avg, 1 from channel-wise max
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolutional weights."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Spatially-scaled feature tensor of shape (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(concat))
        return x * spatial_weights


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining Channel and Spatial attention."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7) -> None:
        """Initialize CBAM.

        Args:
            in_channels: Number of incoming feature channels.
            reduction_ratio: Reduction factor for Channel Attention (default: 16).
            kernel_size: Spatial attention kernel size (default: 7).
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass sequentially applying channel and spatial attention.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attended feature tensor preserving shape (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SnakeAttentionNet(nn.Module):
    """Attention-augmented ResNet architecture for snake venomous classification.

    Applies CBAM after each residual stage (layer1 through layer4), followed by
    Global Average Pooling, Dropout, and a Linear classification head.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        attention: str = "cbam",
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        """Initialize SnakeAttentionNet.

        Args:
            backbone: Backbone architecture ('resnet50' or 'resnet18').
            pretrained: Whether to load ImageNet pretrained weights.
            attention: Attention type ('cbam' or 'none').
            dropout: Dropout probability before the classification head.
            num_classes: Number of output prediction classes.
        """
        super().__init__()
        self.backbone_name = backbone.lower()
        self.use_attention = (attention.lower() == "cbam")
        self.num_classes = num_classes

        # Load backbone with modern torchvision weights API and graceful offline fallback
        base_model, channels = self._create_backbone(self.backbone_name, pretrained)

        # Stem layers (conv1, bn1, relu, maxpool)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Residual stages
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Freshly initialized CBAM attention modules for each residual stage
        if self.use_attention:
            self.cbam1 = CBAM(channels[0])
            self.cbam2 = CBAM(channels[1])
            self.cbam3 = CBAM(channels[2])
            self.cbam4 = CBAM(channels[3])
            # Expose last convolutional feature map target layer for Grad-CAM
            self.target_layer: nn.Module = self.cbam4
        else:
            self.cbam1 = nn.Identity()
            self.cbam2 = nn.Identity()
            self.cbam3 = nn.Identity()
            self.cbam4 = nn.Identity()
            self.target_layer: nn.Module = self.layer4

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[3], num_classes)

        # Initialize classification head
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    @staticmethod
    def _create_backbone(
        backbone_name: str, pretrained: bool
    ) -> tuple[nn.Module, list[int]]:
        """Instantiate backbone and return module and stage channel dimensions."""
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            try:
                base_model = resnet50(weights=weights)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Unable to download or load pretrained weights for resnet50 ({e}). "
                    "Falling back to random weight initialization."
                )
                base_model = resnet50(weights=None)
            channels = [256, 512, 1024, 2048]

        elif backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            try:
                base_model = resnet18(weights=weights)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Unable to download or load pretrained weights for resnet18 ({e}). "
                    "Falling back to random weight initialization."
                )
                base_model = resnet18(weights=None)
            channels = [64, 128, 256, 512]

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                "Supported options are: ['resnet50', 'resnet18']."
            )

        return base_model, channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stem, residual stages + CBAM, pooling, and classification head.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes).
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage 1 + Attention
        x = self.layer1(x)
        x = self.cbam1(x)

        # Stage 2 + Attention
        x = self.layer2(x)
        x = self.cbam2(x)

        # Stage 3 + Attention
        x = self.layer3(x)
        x = self.cbam3(x)

        # Stage 4 + Attention (Grad-CAM target layer)
        x = self.layer4(x)
        x = self.cbam4(x)

        # Pooling & Classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def freeze_backbone(self) -> None:
        """Freeze feature-extraction backbone parameters while keeping CBAM and head trainable."""
        backbone_modules = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = False

        # Explicitly ensure CBAM and classifier head remain trainable
        attention_and_head_modules = [
            self.cbam1,
            self.cbam2,
            self.cbam3,
            self.cbam4,
            self.fc,
        ]
        for module in attention_and_head_modules:
            for param in module.parameters():
                param.requires_grad = True

        logger.info("Backbone frozen: only CBAM modules and classification head are trainable.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the entire network for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen: all network parameters are trainable.")

    def count_parameters(self) -> dict[str, int]:
        """Count total and currently trainable parameters.

        Returns:
            Dict[str, int]: Dictionary with 'total' and 'trainable' parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_target_layer(self) -> nn.Module:
        """Return the target convolutional feature layer for Grad-CAM explainability.

        Returns:
            nn.Module: The final attended convolutional layer (cbam4).
        """
        return self.target_layer


def build_model(cfg: Config | ModelConfig) -> SnakeAttentionNet:
    """Factory function to build and configure the SnakeAttentionNet from configuration.

    Args:
        cfg: Master Config or ModelConfig dataclass instance.

    Returns:
        SnakeAttentionNet: Initialized neural network ready for training or inference.
    """
    model_cfg = cfg.model if isinstance(cfg, Config) else cfg

    model = SnakeAttentionNet(
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        attention=model_cfg.attention,
        dropout=model_cfg.dropout,
        num_classes=model_cfg.num_classes,
    )
    return model
