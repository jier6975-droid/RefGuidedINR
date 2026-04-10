"""
Data loader for RefGuidedINR.

Expects the following directory structure:
    <split>/
        LR/       — low-resolution query images
        RefHR/    — reference high-resolution images
        GT/       — ground-truth high-resolution images

All three images in a triplet must share the same filename.
At training time images are randomly cropped to patch_size × patch_size
(at LR resolution) and scaled to a random HR resolution in [scale_min, scale_max].
"""
import os
import random
import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_images(folder: str) -> list:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    names = sorted(
        f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts
    )
    return names


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


_to_tensor = transforms.ToTensor()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RefGuidedDataset(Dataset):
    """
    PyTorch Dataset for (LR, RefHR, GT) image triplets.

    Args:
        root_dir: Path to the split directory containing LR/, RefHR/, GT/ sub-folders.
        patch_size: Spatial size of the LR patch to crop during training.
                    Set to None to use the full image (evaluation mode).
        scale_min: Minimum up-scaling factor for the HR coordinate grid.
        scale_max: Maximum up-scaling factor for the HR coordinate grid.
        augment: Whether to apply random horizontal/vertical flips.
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: Optional[int] = 48,
        scale_min: float = 1.0,
        scale_max: float = 4.0,
        augment: bool = True,
    ):
        self.lr_dir = os.path.join(root_dir, "LR")
        self.ref_dir = os.path.join(root_dir, "RefHR")
        self.gt_dir = os.path.join(root_dir, "GT")
        self.patch_size = patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment

        self.filenames = _list_images(self.lr_dir)
        assert len(self.filenames) > 0, f"No images found in {self.lr_dir}"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        name = self.filenames[idx]
        lr = _load_image(os.path.join(self.lr_dir, name))
        ref = _load_image(os.path.join(self.ref_dir, name))
        gt = _load_image(os.path.join(self.gt_dir, name))

        # Determine HR size from GT
        gt_w, gt_h = gt.size
        lr_w, lr_h = lr.size

        if self.patch_size is not None:
            # Random crop on LR; apply corresponding crop to GT
            ps = min(self.patch_size, lr_h, lr_w)
            x0 = random.randint(0, lr_w - ps)
            y0 = random.randint(0, lr_h - ps)

            # Compute corresponding GT crop region
            scale_x = gt_w / lr_w
            scale_y = gt_h / lr_h
            gx0 = int(x0 * scale_x)
            gy0 = int(y0 * scale_y)
            gx1 = gx0 + int(ps * scale_x)
            gy1 = gy0 + int(ps * scale_y)

            lr = lr.crop((x0, y0, x0 + ps, y0 + ps))
            gt = gt.crop((gx0, gy0, gx1, gy1))

            # Pick a random scale and resize GT to that size
            scale = random.uniform(self.scale_min, self.scale_max)
            hr_h = max(1, round(ps * scale))
            hr_w = max(1, round(ps * scale))
            gt = gt.resize((hr_w, hr_h), Image.BICUBIC)
        else:
            scale = gt_h / lr_h  # use full scale from GT

        # Optional data augmentation
        if self.augment:
            if random.random() > 0.5:
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
                ref = ref.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
                ref = ref.transpose(Image.FLIP_TOP_BOTTOM)
                gt = gt.transpose(Image.FLIP_TOP_BOTTOM)

        lr_t = _to_tensor(lr)   # (3, H_lr, W_lr)
        ref_t = _to_tensor(ref)  # (3, H_ref, W_ref)
        gt_t = _to_tensor(gt)   # (3, H_hr, W_hr)

        # Build coordinate grid and cell for GT image
        coords, cell = _make_coord_cell(gt_t.shape[-2], gt_t.shape[-1])

        # Flatten GT to (N, 3) for per-pixel supervision
        gt_flat = gt_t.view(3, -1).permute(1, 0)  # (H*W, 3)

        return {
            "lr": lr_t,
            "ref": ref_t,
            "gt": gt_flat,
            "coords": coords,
            "cell": cell,
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Coordinate/cell helpers
# ---------------------------------------------------------------------------

def _make_coord_cell(
    h: int, w: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a grid of normalised 2D coordinates and corresponding cell sizes
    for an image of size (h, w).

    Returns:
        coords: (H*W, 2) in range [-1, 1] (x, y order matching grid_sample).
        cell: (H*W, 2) representing the normalised width and height of each cell.
    """
    rows = torch.linspace(-1 + 1 / h, 1 - 1 / h, h)
    cols = torch.linspace(-1 + 1 / w, 1 - 1 / w, w)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # (H, W) each
    coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W, 2)

    cell_h = torch.full((h * w,), 2.0 / h)
    cell_w = torch.full((h * w,), 2.0 / w)
    cell = torch.stack([cell_w, cell_h], dim=-1)  # (H*W, 2)

    return coords, cell


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_data_loader(
    cfg: dict,
    split: str = "train",
) -> DataLoader:
    """
    Build a DataLoader from the configuration dict.

    Args:
        cfg: Parsed YAML config dict.
        split: 'train' or 'val'.

    Returns:
        torch.utils.data.DataLoader instance.
    """
    data_cfg = cfg.get("data", {})
    root_dir = data_cfg.get(f"{split}_dir", f"datasets/{split}")
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    patch_size = data_cfg.get("patch_size", 48) if split == "train" else None
    scale_min = data_cfg.get("scale_min", 1.0)
    scale_max = data_cfg.get("scale_max", 4.0)
    augment = data_cfg.get("augment", True) if split == "train" else False

    dataset = RefGuidedDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        scale_min=scale_min,
        scale_max=scale_max,
        augment=augment,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader
