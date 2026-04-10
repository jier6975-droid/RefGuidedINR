"""
Evaluation metrics for super-resolution: PSNR, SSIM, and LPIPS.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image tensor, values in [0, max_val].
        target: Ground-truth image tensor, values in [0, max_val].
        max_val: Maximum possible pixel value.

    Returns:
        PSNR in dB (float).
    """
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index Measure (mean over channels).

    Args:
        pred: (B, C, H, W) or (C, H, W) predicted tensor.
        target: (B, C, H, W) or (C, H, W) GT tensor.
        data_range: Value range of inputs (default 1.0 for [0,1] images).

    Returns:
        Mean SSIM value (float).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    pred = pred.float()
    target = target.float()
    B, C, H, W = pred.shape

    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model: Optional[object] = None


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    Args:
        pred: (B, 3, H, W) predicted image, values in [0, 1].
        target: (B, 3, H, W) GT image, values in [0, 1].
        net: Backbone network for LPIPS ('alex', 'vgg', or 'squeeze').

    Returns:
        Mean LPIPS score (float, lower is better).
    """
    global _lpips_model
    try:
        import lpips as lpips_lib
        if _lpips_model is None:
            _lpips_model = lpips_lib.LPIPS(net=net)
            _lpips_model.eval()
        device = pred.device
        _lpips_model = _lpips_model.to(device)
        # LPIPS expects inputs in [-1, 1]
        pred_norm = pred.float() * 2 - 1
        target_norm = target.float() * 2 - 1
        with torch.no_grad():
            score = _lpips_model(pred_norm, target_norm)
        return score.mean().item()
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks the running mean of a scalar metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: {self.avg:.4f})"
