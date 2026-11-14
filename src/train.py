"""Training pipeline with two-phase fine-tuning, mixed precision, and early stopping.

Usage:
    python -m src.train --config configs/config.yaml [--data_dir PATH] [--epochs N] [--resume PATH]
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from src.config import Config, load_config
from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    LABEL_VENOMOUS,
    get_dataloaders,
)
from src.model import SnakeAttentionNet, build_model
from src.utils import ensure_dirs, get_device, set_seed, setup_logger


def get_parameter_groups(
    model: SnakeAttentionNet, base_lr: float, weight_decay: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate backbone parameters from attention and classification head parameters.

    Returns:
        Tuple[List[Dict], List[Dict]]:
            - phase1_groups: only head and CBAM parameters (for warm-up)
            - phase2_groups: differential learning rates (backbone=lr*0.1, head=lr)
    """
    backbone_modules = [
        model.conv1,
        model.bn1,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]
    attention_and_head_modules = [
        model.cbam1,
        model.cbam2,
        model.cbam3,
        model.cbam4,
        model.fc,
    ]

    backbone_params = []
    backbone_param_ids = set()
    for m in backbone_modules:
        for p in m.parameters():
            if id(p) not in backbone_param_ids:
                backbone_param_ids.add(id(p))
                backbone_params.append(p)

    head_params = []
    head_param_ids = set()
    for m in attention_and_head_modules:
        for p in m.parameters():
            if id(p) not in backbone_param_ids and id(p) not in head_param_ids:
                head_param_ids.add(id(p))
                head_params.append(p)

    phase1_groups = [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay}
    ]

    phase2_groups = [
        {
            "params": backbone_params,
            "lr": base_lr * 0.1,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": head_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "head",
        },
    ]

    return phase1_groups, phase2_groups


def plot_training_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    """Save training curves plot for loss and accuracy."""
    if not history:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs, val_loss, "s-", label="Val Loss", color="crimson", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, train_acc, "o-", label="Train Acc", color="royalblue", lw=2)
    ax2.plot(epochs, val_acc, "s-", label="Val Acc", color="crimson", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Classification Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history: list[dict[str, Any]], csv_path: Path) -> None:
    """Save epoch history list to CSV file."""
    if not history:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_venomous_prec",
        "val_venomous_recall",
        "val_venomous_f1",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            filtered_row = {k: row.get(k, 0) for k in fieldnames}
            writer.writerow(filtered_row)


def train_model(
    cfg: Config,
    data_dir: str | None = None,
    override_epochs: int | None = None,
    resume_path: str | None = None,
) -> None:
    """Run full two-phase model training pipeline.

    Args:
        cfg: Configuration dataclass.
        data_dir: Optional CLI override for raw dataset directory.
        override_epochs: Optional CLI override for total training epochs.
        resume_path: Optional path to checkpoint to resume training from.
    """
    effective_data_dir = Path(data_dir) if data_dir is not None else cfg.paths.raw_dir
    total_epochs = override_epochs if override_epochs is not None else cfg.train.epochs
    freeze_epochs = min(cfg.train.freeze_backbone_epochs, total_epochs)

    # Ensure directories
    ensure_dirs(
        cfg.paths.checkpoint_dir,
        cfg.paths.plots_dir,
        cfg.paths.logs_dir,
        cfg.paths.processed_dir,
    )

    logger = setup_logger("snake_train", log_file=cfg.paths.logs_dir / "train.log")
    logger.info("=" * 70)
    logger.info("STARTING TRAINING PIPELINE: Snake Venomous Detection")
    logger.info(f"Dataset root: {effective_data_dir.resolve()}")
    logger.info(f"Total epochs: {total_epochs} (Backbone frozen for first {freeze_epochs} epochs)")
    logger.info("=" * 70)

    # Set seed & device
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Target computational device: {device.type.upper()}")

    # DataLoaders (Test set is intentionally unread in this file to prevent data leakage)
    train_loader, val_loader, _, class_weights = get_dataloaders(
        cfg, raw_dir=effective_data_dir
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = build_model(cfg).to(device)

    # Loss function with class weights and label smoothing
    loss_weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=cfg.train.label_smoothing,
    )
    if loss_weights is not None:
        logger.info(f"Using class-weighted CrossEntropyLoss: {loss_weights.tolist()}")

    # Parameter groups
    phase1_groups, phase2_groups = get_parameter_groups(
        model, base_lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # Mixed precision setup (CUDA only)
    use_amp = (device.type == "cuda" and cfg.train.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    logger.info(f"Mixed precision (AMP) enabled: {use_amp}")

    start_epoch = 1
    best_score = -1.0
    patience_counter = 0
    history: list[dict[str, Any]] = []
    is_in_phase2 = False

    # Check resume
    if resume_path:
        res_file = Path(resume_path)
        if not res_file.is_file():
            raise FileNotFoundError(f"Resume checkpoint '{res_file.resolve()}' not found.")

        logger.info(f"Resuming training from checkpoint: {res_file.resolve()}")
        checkpoint = torch.load(res_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", -1.0)
        patience_counter = checkpoint.get("patience_counter", 0)
        history = checkpoint.get("history", [])

        # Determine phase upon resumption
        if start_epoch > freeze_epochs:
            is_in_phase2 = True
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(phase2_groups)
        else:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(phase1_groups)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        remaining = max(1, total_epochs - start_epoch + 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=cfg.train.lr * 0.01
        )
        if checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load scheduler state ({e}); using fresh scheduler.")
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed successfully at epoch {start_epoch} (best score so far: {best_score:.4f})")
    else:
        # Phase 1 initialization (warm-up with frozen backbone)
        if freeze_epochs > 0:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(phase1_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=freeze_epochs, eta_min=cfg.train.lr * 0.1
            )
            logger.info("Initialized Phase 1: Backbone frozen (training CBAM + classification head).")
        else:
            is_in_phase2 = True
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(phase2_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=cfg.train.lr * 0.01
            )
            logger.info("Initialized Phase 2 immediately: Full network training.")

    # Checkpoint output paths
    best_ckpt_path = cfg.paths.checkpoint_dir / "best.pt"
    last_ckpt_path = cfg.paths.checkpoint_dir / "last.pt"
    history_csv_path = cfg.paths.logs_dir / "history.csv"
    curves_plot_path = cfg.paths.plots_dir / "training_curves.png"

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            # Phase transition: Unfreeze backbone after freeze_epochs
            if epoch > freeze_epochs and not is_in_phase2:
                logger.info(f"--- Entering Phase 2 at epoch {epoch}: Unfreezing backbone with lr={cfg.train.lr * 0.1} ---")
                model.unfreeze_backbone()
                # Rebuild optimizer with 2 parameter groups (differential lr)
                optimizer = torch.optim.AdamW(phase2_groups)
                remaining_epochs = max(1, total_epochs - epoch + 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining_epochs, eta_min=cfg.train.lr * 0.01
                )
                is_in_phase2 = True

            # -------------------------------------------------------------
            # Training Loop
            # -------------------------------------------------------------
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False, ncols=100)
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                    if torch.isnan(loss):
                        raise RuntimeError(f"NaN loss encountered at epoch {epoch}!")

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    if torch.isnan(loss):
                        raise RuntimeError(f"NaN loss encountered at epoch {epoch}!")

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                batch_size = targets.size(0)
                train_loss_sum += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += batch_size

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = train_loss_sum / train_total
            train_acc = (train_correct / train_total) * 100.0

            # Step scheduler
            current_lr = optimizer.param_groups[-1]["lr"]
            scheduler.step()

            # -------------------------------------------------------------
            # Validation Loop
            # -------------------------------------------------------------
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            all_preds: list[int] = []
            all_targets: list[int] = []

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    if use_amp:
                        with torch.amp.autocast(device_type=device.type, enabled=True):
                            outputs = model(images)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                    batch_size = targets.size(0)
                    val_loss_sum += loss.item() * batch_size
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += batch_size

                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(targets.cpu().tolist())

            val_loss = val_loss_sum / val_total
            val_acc = (val_correct / val_total) * 100.0

            # Venomous class (class 1) precision, recall, F1
            tp = sum(1 for p, t in zip(all_preds, all_targets) if p == LABEL_VENOMOUS and t == LABEL_VENOMOUS)
            fp = sum(1 for p, t in zip(all_preds, all_targets) if p == LABEL_VENOMOUS and t != LABEL_VENOMOUS)
            fn = sum(1 for p, t in zip(all_preds, all_targets) if p != LABEL_VENOMOUS and t == LABEL_VENOMOUS)

            val_prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            val_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            val_f1 = (2 * val_prec * val_recall / (val_prec + val_recall)) if (val_prec + val_recall) > 0 else 0.0

            # Log epoch summary
            logger.info(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"Venomous Prec: {val_prec:.3f} Rec: {val_recall:.3f} F1: {val_f1:.3f} | "
                f"LR: {current_lr:.6f}"
            )

            # Record history
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_venomous_prec": val_prec,
                "val_venomous_recall": val_recall,
                "val_venomous_f1": val_f1,
                "lr": current_lr,
            }
            history.append(epoch_data)

            # -------------------------------------------------------------
            # Early Stopping & Checkpoints
            # -------------------------------------------------------------
            # Monitored rule: Validation Venomous F1 (safety-critical medical metric)
            score = val_f1
            improved = score > best_score + 1e-4

            if improved:
                best_score = score
                patience_counter = 0
                logger.info(f"  --> Best score improved to {best_score:.4f}! Saving checkpoint to {best_ckpt_path.name}")

                best_payload = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "monitored_metric": "val_venomous_f1",
                    "best_score": best_score,
                    "val_metrics": {
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_precision": val_prec,
                        "val_recall": val_recall,
                        "val_f1": val_f1,
                    },
                    "class_to_idx": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                    "class_names": CLASS_NAMES,
                    "img_size": cfg.data.img_size,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                    "backbone": cfg.model.backbone,
                    "attention": cfg.model.attention,
                }
                torch.save(best_payload, best_ckpt_path)
            else:
                patience_counter += 1
                logger.info(f"  --> No score improvement ({patience_counter}/{cfg.train.patience})")

            # Always save last.pt for uninterrupted resume capability
            last_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_score": best_score,
                "patience_counter": patience_counter,
                "history": history,
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
                "class_names": CLASS_NAMES,
                "img_size": cfg.data.img_size,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "backbone": cfg.model.backbone,
                "attention": cfg.model.attention,
            }
            torch.save(last_payload, last_ckpt_path)

            # Persist history and plots
            save_history_csv(history, history_csv_path)
            plot_training_curves(history, curves_plot_path)

            # Early stopping check
            if patience_counter >= cfg.train.patience:
                logger.info(f"Early stopping triggered after {cfg.train.patience} epochs without improvement.")
                break

    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPTED] KeyboardInterrupt received. Saving current checkpoint state...")
        last_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_score": best_score,
            "patience_counter": patience_counter,
            "history": history,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
            "class_names": CLASS_NAMES,
            "img_size": cfg.data.img_size,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "backbone": cfg.model.backbone,
            "attention": cfg.model.attention,
        }
        torch.save(last_payload, last_ckpt_path)
        save_history_csv(history, history_csv_path)
        plot_training_curves(history, curves_plot_path)
        logger.info(f"Emergency training checkpoint saved cleanly to '{last_ckpt_path}'. Exiting.")
        return

    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE. Best Venomous F1: {best_score:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt_path.resolve()}")
    logger.info(f"Last checkpoint: {last_ckpt_path.resolve()}")
    logger.info(f"Training history saved: {history_csv_path.resolve()}")
    logger.info(f"Curves plot saved: {curves_plot_path.resolve()}")
    logger.info("=" * 70)


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train Snake Attention Network for Venomous Classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override raw data directory path (e.g. data/raw_smoke)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total number of training epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (e.g. outputs/checkpoints/last.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_model(
        cfg=cfg,
        data_dir=args.data_dir,
        override_epochs=args.epochs,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
