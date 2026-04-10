"""
Utility helpers: config loading, reproducibility, and coordinate utilities.
"""
import os
import random
from typing import Tuple

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_coord(shape: Tuple[int, int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Build a normalised coordinate grid for an image of the given (H, W) shape.

    Returns:
        coords: (H*W, 2) tensor, x-y order, values in [-1, 1].
    """
    h, w = shape
    rows = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    cols = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    return coords


def to_pixel_samples(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an image tensor to (coords, values) pairs for INR supervision.

    Args:
        img: (C, H, W) image tensor.

    Returns:
        coords: (H*W, 2) normalised coordinate grid.
        values: (H*W, C) flattened pixel values.
    """
    _, h, w = img.shape
    coords = make_coord((h, w))
    values = img.view(img.shape[0], -1).permute(1, 0)  # (H*W, C)
    return coords, values


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = "checkpoint.pth",
) -> None:
    """Save a training checkpoint to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint from disk."""
    return torch.load(path, map_location=device)
