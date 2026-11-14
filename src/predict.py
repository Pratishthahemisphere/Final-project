"""Production inference engine with uncertainty thresholding and EXIF orientation handling.

Usage:
    python -m src.predict --image PATH_OR_DIR [--checkpoint PATH] [--threshold FLOAT]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps

from src.dataset import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    LABEL_NON_VENOMOUS,
    LABEL_VENOMOUS,
)
from src.model import SnakeAttentionNet
from src.utils import get_device

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_preprocess_pil(image_source: Image.Image | Path | str) -> Image.Image:
    """Load image safely, transpose according to EXIF orientation, and convert to standard RGB.

    Handles non-RGB formats (RGBA, grayscale, palette, CMYK) gracefully.

    Args:
        image_source: PIL Image instance or path to an image file.

    Returns:
        Image.Image: Standard 3-channel RGB PIL image.

    Raises:
        FileNotFoundError: If image file path does not exist.
        ValueError: If file is corrupted or unreadable.
        TypeError: If input is not a recognized type.
    """
    if isinstance(image_source, (str, Path)):
        img_path = Path(image_source)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: '{img_path.resolve()}'")
        try:
            with Image.open(img_path) as raw_img:
                img = ImageOps.exif_transpose(raw_img)
                return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to read image at '{img_path}': {e}") from e

    elif isinstance(image_source, Image.Image):
        img = ImageOps.exif_transpose(image_source)
        return img.convert("RGB")

    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_source)}. Expected PIL Image, str, or Path."
        )


class SnakeClassifier:
    """Production-grade classifier for snake identification with safety-critical uncertainty gating."""

    def __init__(
        self,
        checkpoint_path: str | Path = "outputs/checkpoints/best.pt",
        device: torch.device | None = None,
        uncertainty_threshold: float | None = None,
    ) -> None:
        """Initialize SnakeClassifier by loading model weights and normalization stats.

        Args:
            checkpoint_path: Path to trained PyTorch checkpoint (.pt).
            device: Computation device (auto-detected if None).
            uncertainty_threshold: Minimum confidence threshold to return a definitive label.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file '{self.checkpoint_path.resolve()}' not found. "
                "Ensure training has finished and saved outputs/checkpoints/best.pt."
            )

        self.device = device if device is not None else get_device()
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Restore architecture hyperparameters
        backbone = self.checkpoint.get("backbone", "resnet50")
        attention = self.checkpoint.get("attention", "cbam")
        num_classes = len(self.checkpoint.get("class_names", CLASS_NAMES))

        self.model = SnakeAttentionNet(
            backbone=backbone,
            pretrained=False,
            attention=attention,
            dropout=0.3,
            num_classes=num_classes,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Restore resolution and normalization parameters
        self.img_size = int(self.checkpoint.get("img_size", 224))
        norm_dict = self.checkpoint.get("normalization", {})
        self.mean = norm_dict.get("mean", [0.485, 0.456, 0.406])
        self.std = norm_dict.get("std", [0.229, 0.224, 0.225])

        # Restore uncertainty threshold
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = float(uncertainty_threshold)
        else:
            cfg_dict = self.checkpoint.get("config", {})
            inf_cfg = cfg_dict.get("inference", {}) if isinstance(cfg_dict, dict) else {}
            self.uncertainty_threshold = float(inf_cfg.get("uncertainty_threshold", 0.70))

        resize_dim = int(self.img_size * 256 / 224)
        self.transform = T.Compose([
            T.Resize(resize_dim),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def predict(
        self,
        image: Image.Image | Path | str,
        uncertainty_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform classification on a single image.

        Args:
            image: PIL Image or path to image.
            uncertainty_threshold: Optional threshold override for this call.

        Returns:
            Dict[str, Any]:
                - 'label': 'venomous', 'non_venomous', or 'uncertain'
                - 'raw_label': argmax predicted class ('venomous' or 'non_venomous')
                - 'confidence': maximum softmax probability
                - 'uncertain': boolean indicating if confidence < threshold
                - 'uncertainty_threshold': threshold value used
                - 'probabilities': dictionary of probabilities per class
        """
        rgb_img = load_and_preprocess_pil(image)
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        p_non_venomous = float(probs[LABEL_NON_VENOMOUS].item())
        p_venomous = float(probs[LABEL_VENOMOUS].item())

        raw_idx = int(torch.argmax(probs).item())
        raw_label = IDX_TO_CLASS[raw_idx]
        confidence = float(probs[raw_idx].item())

        active_threshold = (
            uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold
        )
        uncertain = bool(confidence < active_threshold)
        final_label = "uncertain" if uncertain else raw_label

        return {
            "label": final_label,
            "raw_label": raw_label,
            "confidence": confidence,
            "uncertain": uncertain,
            "uncertainty_threshold": active_threshold,
            "probabilities": {
                "non_venomous": p_non_venomous,
                "venomous": p_venomous,
            },
        }

    def predict_batch(
        self,
        images: list[Image.Image | Path | str],
        batch_size: int = 16,
        uncertainty_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Perform batched inference on multiple images.

        Args:
            images: List of PIL images or image paths.
            batch_size: Batch size for batched forward passes.
            uncertainty_threshold: Optional threshold override.

        Returns:
            List[Dict[str, Any]]: List of prediction result dictionaries.
        """
        results: list[dict[str, Any]] = []
        active_threshold = (
            uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold
        )

        for i in range(0, len(images), batch_size):
            batch_slice = images[i : i + batch_size]
            tensors = []
            for item in batch_slice:
                rgb_img = load_and_preprocess_pil(item)
                tensors.append(self.transform(rgb_img))

            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            with torch.inference_mode():
                logits = self.model(batch_tensor)
                batch_probs = F.softmax(logits, dim=1)

            for j in range(batch_probs.size(0)):
                probs = batch_probs[j]
                p_nv = float(probs[LABEL_NON_VENOMOUS].item())
                p_v = float(probs[LABEL_VENOMOUS].item())
                raw_idx = int(torch.argmax(probs).item())
                raw_label = IDX_TO_CLASS[raw_idx]
                conf = float(probs[raw_idx].item())
                uncertain = bool(conf < active_threshold)
                final_label = "uncertain" if uncertain else raw_label

                results.append({
                    "label": final_label,
                    "raw_label": raw_label,
                    "confidence": conf,
                    "uncertain": uncertain,
                    "uncertainty_threshold": active_threshold,
                    "probabilities": {
                        "non_venomous": p_nv,
                        "venomous": p_v,
                    },
                })

        return results


def print_prediction_table(results: list[tuple[str, dict[str, Any]]]) -> None:
    """Print formatted CLI table of inference outputs."""
    print("=" * 85)
    header = f"{'Image Path / Source':<32} | {'Predicted':<12} | {'Conf':<7} | {'P(Venomous)':<11} | {'P(Non-Ven)':<10} | {'Status'}"
    print(header)
    print("-" * 85)

    for path_str, res in results:
        short_path = path_str if len(path_str) <= 30 else f"...{path_str[-27:]}"
        label = res["label"].upper()
        conf = f"{res['confidence'] * 100:.1f}%"
        p_v = f"{res['probabilities']['venomous'] * 100:.1f}%"
        p_nv = f"{res['probabilities']['non_venomous'] * 100:.1f}%"
        status = "UNCERTAIN" if res["uncertain"] else "CONFIDENT"

        print(f"{short_path:<32} | {label:<12} | {conf:<7} | {p_v:<11} | {p_nv:<10} | {status}")

    print("=" * 85)


def main() -> None:
    """Parse CLI arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Predict snake venomous status with safety uncertainty gating."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file or directory of images to classify",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint (default: outputs/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override uncertainty threshold (e.g. 0.70)",
    )
    args = parser.parse_args()

    classifier = SnakeClassifier(
        checkpoint_path=args.checkpoint,
        uncertainty_threshold=args.threshold,
    )

    target_path = Path(args.image)
    image_paths: list[Path] = []

    if target_path.is_file():
        image_paths = [target_path]
    elif target_path.is_dir():
        image_paths = [
            p for p in sorted(target_path.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if not image_paths:
            print(f"No valid image files found in '{target_path.resolve()}'.")
            sys.exit(1)
    else:
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    table_data = []
    for p in image_paths:
        res = classifier.predict(p)
        table_data.append((p.name, res))

    print_prediction_table(table_data)


if __name__ == "__main__":
    main()
