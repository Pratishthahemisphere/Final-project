"""Utility functions for reproducibility, device selection, logging, and filesystem management."""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """Auto-detect and return the optimal computing device following priority: cuda -> mps -> cpu.

    Returns:
        torch.device: PyTorch device object for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "snake_attention",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a structured logger with console and optional file output.

    Args:
        name: Name of the logger.
        log_file: Optional path to output log file.
        level: Logging verbosity level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_dirs(*dirs: str | Path) -> None:
    """Ensure that all provided directory paths exist, creating parent directories if needed.

    Args:
        *dirs: Variable number of directory paths to create.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
