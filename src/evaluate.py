"""Comprehensive evaluation pipeline on the held-out test split.

Usage:
    python -m src.evaluate --config configs/config.yaml [--data_dir PATH] [--checkpoint PATH]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.dataset import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    LABEL_NON_VENOMOUS,
    LABEL_VENOMOUS,
    SnakeDataset,
    get_val_transform,
)
from src.model import SnakeAttentionNet
from src.utils import ensure_dirs, get_device, setup_logger


def resolve_sample_path(path_str: str, override_data_dir: Path | None = None) -> Path:
    """Resolve a sample path string to an existing filesystem Path.

    Args:
        path_str: Path string from splits.json.
        override_data_dir: Optional data directory override.

    Returns:
        Path: Existing path to image.
    """
    p = Path(path_str)
    if p.is_file():
        return p

    cwd_p = Path.cwd() / p
    if cwd_p.is_file():
        return cwd_p

    if override_data_dir:
        # Try matching by class_folder/filename
        cls_sub = p.parent.name
        candidate = override_data_dir / cls_sub / p.name
        if candidate.is_file():
            return candidate
        candidate_flat = override_data_dir / p.name
        if candidate_flat.is_file():
            return candidate_flat

    return p


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save styled confusion matrix plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: list[int], y_scores: list[float], roc_auc: float | None, out_path: Path) -> None:
    """Save ROC curve plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    if roc_auc is not None and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    else:
        ax.text(0.5, 0.5, "ROC undefined (single class)", ha="center", va="center")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Venomous Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: list[int], y_scores: list[float], pr_auc: float | None, out_path: Path) -> None:
    """Save Precision-Recall curve plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    if pr_auc is not None and len(set(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, color="forestgreen", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
        prevalence = sum(y_true) / len(y_true)
        ax.axhline(prevalence, color="gray", linestyle="--", label=f"Prevalence ({prevalence:.1%})")
    else:
        ax.text(0.5, 0.5, "PR undefined (single class)", ha="center", va="center")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Venomous Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_examples(
    misclassified: list[dict[str, Any]],
    mean: list[float],
    std: list[float],
    out_path: Path,
) -> None:
    """Plot grid of up to 12 most confident errors with unnormalized RGB images."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not misclassified:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            "No Misclassified Examples!\n(100% Test Accuracy)",
            ha="center", va="center", fontsize=14, color="forestgreen", weight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Sort by prediction confidence descending
    sorted_errors = sorted(misclassified, key=lambda x: x["confidence"], reverse=True)[:12]
    n_examples = len(sorted_errors)
    cols = min(4, n_examples)
    rows = (n_examples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_examples:
            item = sorted_errors[i]
            # Unnormalize: [C, H, W] -> [H, W, C]
            tensor_np = item["tensor"].numpy().transpose(1, 2, 0)
            img_disp = np.clip(tensor_np * std_arr + mean_arr, 0.0, 1.0)

            ax.imshow(img_disp)
            true_label = IDX_TO_CLASS[item["true_label"]]
            pred_label = IDX_TO_CLASS[item["pred_label"]]
            conf = item["confidence"]

            # Highlight dangerous false-safe errors in red
            color = "red" if (true_label == "venomous" and pred_label == "non_venomous") else "darkorange"
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({conf:.1%})",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle("Top Confidently Misclassified Test Images", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    cfg: Config,
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Execute complete test evaluation pipeline and export all diagnostic artifacts.

    Args:
        cfg: Master Config dataclass.
        data_dir: Optional raw data directory override.
        checkpoint_path: Optional path to checkpoint to evaluate.

    Returns:
        Dict[str, Any]: Comprehensive test metrics dictionary.
    """
    ckpt_file = Path(checkpoint_path) if checkpoint_path else cfg.paths.checkpoint_dir / "best.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint file '{ckpt_file.resolve()}' not found. "
            "Please train a model first using src.train."
        )

    ensure_dirs(cfg.paths.plots_dir, cfg.paths.logs_dir)
    logger = setup_logger("snake_eval", log_file=cfg.paths.logs_dir / "eval.log")

    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE: Held-out Test Split")
    logger.info(f"Checkpoint: {ckpt_file.resolve()}")
    logger.info("=" * 70)

    device = get_device()
    logger.info(f"Using device: {device.type.upper()}")

    # 1. Load Checkpoint & Rebuild Model
    logger.info("Loading checkpoint weights and configuration...")
    ckpt = torch.load(ckpt_file, map_location=device)

    backbone = ckpt.get("backbone", cfg.model.backbone)
    attention = ckpt.get("attention", cfg.model.attention)

    model = SnakeAttentionNet(
        backbone=backbone,
        pretrained=False,  # Loaded strictly from checkpoint
        attention=attention,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Load Held-Out Test Split from data/processed/splits.json
    splits_file = Path(cfg.paths.processed_dir) / "splits.json"
    if not splits_file.is_file():
        raise FileNotFoundError(
            f"Splits file not found at '{splits_file.resolve()}'. "
            "Run training or build_splits to establish the deterministic splits first."
        )

    with splits_file.open("r", encoding="utf-8") as sf:
        splits_data = json.load(sf)

    raw_test_samples = splits_data.get("test", [])
    if not raw_test_samples:
        logger.error("[FATAL] Test split in splits.json is empty!")
        return {}

    override_dir = Path(data_dir) if data_dir else None
    resolved_test_samples = [
        (resolve_sample_path(p_str, override_dir), int(lbl))
        for p_str, lbl in raw_test_samples
    ]

    # Verify files exist
    valid_test_samples = [(p, lbl) for p, lbl in resolved_test_samples if p.is_file()]
    if not valid_test_samples:
        logger.error("[FATAL] None of the test sample files could be located on disk!")
        return {}

    logger.info(f"Loaded {len(valid_test_samples)} test samples strictly from test split.")

    val_transform = get_val_transform(cfg.data.img_size)
    test_dataset = SnakeDataset(valid_test_samples, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Model Inference (torch.inference_mode)
    all_targets: list[int] = []
    all_probs: list[float] = []      # Probability of venomous (class 1)
    all_preds: list[int] = []        # Predicted class at default 0.5 threshold
    all_confidences: list[float] = [] # Max softmax probability
    misclassified_items: list[dict[str, Any]] = []

    with torch.inference_mode():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            for i in range(targets.size(0)):
                target = int(targets[i].item())
                p_venomous = float(probs[i, LABEL_VENOMOUS].item())
                pred = int(torch.argmax(probs[i]).item())
                conf = float(probs[i, pred].item())

                all_targets.append(target)
                all_probs.append(p_venomous)
                all_preds.append(pred)
                all_confidences.append(conf)

                if pred != target:
                    misclassified_items.append({
                        "tensor": images[i].cpu(),
                        "true_label": target,
                        "pred_label": pred,
                        "confidence": conf,
                        "prob_venomous": p_venomous,
                    })

    # Check for single-class test set
    unique_classes = set(all_targets)
    if len(unique_classes) < 2:
        logger.warning(
            f"[NOTICE] Test set contains only a single class ({unique_classes}). "
            "ROC-AUC and PR-AUC require both classes and will be set to None."
        )

    # 4. Metric Computations
    acc = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    cm = confusion_matrix(all_targets, all_preds, labels=[LABEL_NON_VENOMOUS, LABEL_VENOMOUS])

    tn, fp, fn, tp = cm.ravel()
    venomous_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_safe_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[LABEL_NON_VENOMOUS, LABEL_VENOMOUS], zero_division=0
    )

    roc_auc = float(roc_auc_score(all_targets, all_probs)) if len(unique_classes) > 1 else None
    pr_auc = float(average_precision_score(all_targets, all_probs)) if len(unique_classes) > 1 else None

    # Print Formatted Evaluation Report
    print("\n" + "=" * 70)
    print("               TEST SET PERFORMANCE EVALUATION REPORT")
    print("=" * 70)
    print(f"Overall Accuracy:          {acc * 100:.2f}%")
    print(f"Balanced Accuracy:         {bal_acc * 100:.2f}%")
    print(f"ROC-AUC Score:             {f'{roc_auc:.4f}' if roc_auc is not None else 'N/A'}")
    print(f"PR-AUC (Average Prec):     {f'{pr_auc:.4f}' if pr_auc is not None else 'N/A'}")
    print("-" * 70)
    print("SAFETY-CRITICAL CLINICAL METRICS:")
    print(f"  * Venomous Recall (Sensitivity): {venomous_recall * 100:.2f}%")
    print(f"  * False-Safe Rate (DANGEROUS):   {false_safe_rate * 100:.2f}%  (Venomous mistaken as Safe)")
    print("-" * 70)
    print("PER-CLASS METRICS:")
    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"  - {name:<14} | Precision: {prec_arr[idx]:.4f} | "
            f"Recall: {rec_arr[idx]:.4f} | F1-Score: {f1_arr[idx]:.4f}"
        )
    print("-" * 70)
    print("CONFUSION MATRIX:")
    print("                Predicted Non-Venomous   Predicted Venomous")
    print(f"  True Non-Venomous:     {tn:<24} {fp}")
    print(f"  True Venomous:         {fn:<24} {tp}  <-- (fn={fn} False-Safe!)")
    print("=" * 70)

    # 5. Threshold Analysis Sweep (0.1 to 0.9)
    thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]
    threshold_rows: list[dict[str, Any]] = []
    qualifying_thresholds: list[dict[str, Any]] = []

    for thresh in thresholds:
        sweep_preds = [1 if p >= thresh else 0 for p in all_probs]
        s_cm = confusion_matrix(all_targets, sweep_preds, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = s_cm.ravel()

        s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_fsr = s_fn / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_f1 = (2 * s_prec * s_rec / (s_prec + s_rec)) if (s_prec + s_rec) > 0 else 0.0
        s_acc = (s_tp + s_tn) / len(all_targets) if len(all_targets) > 0 else 0.0

        row = {
            "threshold": thresh,
            "venomous_recall": round(s_rec, 4),
            "false_safe_rate": round(s_fsr, 4),
            "venomous_precision": round(s_prec, 4),
            "venomous_f1": round(s_f1, 4),
            "accuracy": round(s_acc, 4),
            "tp": int(s_tp),
            "fp": int(s_fp),
            "fn": int(s_fn),
            "tn": int(s_tn),
        }
        threshold_rows.append(row)

        if s_rec >= 0.95:
            qualifying_thresholds.append(row)

    # Save threshold analysis CSV
    threshold_csv_path = cfg.paths.logs_dir / "threshold_analysis.csv"
    with threshold_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)
    logger.info(f"Saved threshold sweep analysis to: {threshold_csv_path.resolve()}")

    # Find optimal threshold achieving >= 95% recall with best precision
    print("\nTHRESHOLD ANALYSIS FOR P(venomous):")
    if qualifying_thresholds:
        # Pick highest precision among those with recall >= 95%
        best_thresh_candidate = max(qualifying_thresholds, key=lambda x: (x["venomous_precision"], x["venomous_f1"]))
        print(
            f"  Optimal safety threshold achieving >=95% Venomous Recall: "
            f"T = {best_thresh_candidate['threshold']:.2f}\n"
            f"  --> Recall: {best_thresh_candidate['venomous_recall'] * 100:.1f}%, "
            f"Precision: {best_thresh_candidate['venomous_precision'] * 100:.1f}%, "
            f"False-Safe Rate: {best_thresh_candidate['false_safe_rate'] * 100:.1f}%"
        )
        optimal_threshold_info = best_thresh_candidate
    else:
        # Honestly report if unreachable
        max_rec_thresh = max(threshold_rows, key=lambda x: x["venomous_recall"])
        print(
            f"  [NOTICE] No threshold achieved >=95% venomous recall on this test set.\n"
            f"  Maximum achievable recall was {max_rec_thresh['venomous_recall'] * 100:.1f}% "
            f"at threshold T = {max_rec_thresh['threshold']:.2f}."
        )
        optimal_threshold_info = {"status": "unachievable", "best_attempt": max_rec_thresh}

    # 6. Generate and Save Plots
    cm_path = cfg.paths.plots_dir / "confusion_matrix.png"
    roc_path = cfg.paths.plots_dir / "roc_curve.png"
    pr_path = cfg.paths.plots_dir / "pr_curve.png"
    errors_path = cfg.paths.plots_dir / "misclassified_examples.png"

    plot_confusion_matrix(cm, cm_path)
    plot_roc_curve(all_targets, all_probs, roc_auc, roc_path)
    plot_pr_curve(all_targets, all_probs, pr_auc, pr_path)

    norm_stats = ckpt.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    plot_misclassified_examples(misclassified_items, norm_stats["mean"], norm_stats["std"], errors_path)

    logger.info(f"Saved confusion matrix: {cm_path.resolve()}")
    logger.info(f"Saved ROC curve: {roc_path.resolve()}")
    logger.info(f"Saved PR curve: {pr_path.resolve()}")
    logger.info(f"Saved misclassified grid: {errors_path.resolve()}")

    # 7. Save Metrics JSON
    metrics_json_path = cfg.paths.logs_dir / "test_metrics.json"
    metrics_data = {
        "checkpoint": str(ckpt_file.resolve()),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "venomous_recall": venomous_recall,
        "false_safe_rate": false_safe_rate,
        "class_metrics": {
            CLASS_NAMES[0]: {
                "precision": float(prec_arr[0]),
                "recall": float(rec_arr[0]),
                "f1": float(f1_arr[0]),
            },
            CLASS_NAMES[1]: {
                "precision": float(prec_arr[1]),
                "recall": float(rec_arr[1]),
                "f1": float(f1_arr[1]),
            },
        },
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(all_targets),
        "total_misclassified": len(misclassified_items),
        "optimal_threshold_analysis": optimal_threshold_info,
    }

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Saved test metrics JSON: {metrics_json_path.resolve()}")

    return metrics_data


def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Snake Attention model on held-out test split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional override for raw dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to checkpoint (default: outputs/checkpoints/best.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_model(
        cfg=cfg,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
