"""
train.py — RefGuidedINR Training Pipeline.

Usage:
    python trainers/train.py --config configs/example_config.yaml [--seed 42] [--gpu 0]

The script:
  1. Loads config from YAML.
  2. Builds model, data loaders, optimizer, scheduler, and loss.
  3. Trains for the specified number of epochs, logging to TensorBoard.
  4. Evaluates on the validation set periodically and saves checkpoints.
"""
import sys
import os

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import RefGuidedINR
from data import get_data_loader
from utils import (
    load_config,
    set_seed,
    make_coord,
    save_checkpoint,
    load_checkpoint,
    CombinedLoss,
    compute_psnr,
    compute_ssim,
    AverageMeter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer, cfg: dict, num_steps: int):
    train_cfg = cfg.get("train", {})
    name = train_cfg.get("lr_scheduler", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    if name == "step":
        step_size = train_cfg.get("lr_step", 30)
        gamma = train_cfg.get("lr_gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
    scheduler,
    writer: SummaryWriter,
    epoch: int,
    total_steps: int,
) -> int:
    model.train()
    meters = {k: AverageMeter(k) for k in ["total", "reconstruction", "identity", "perceptual"]}

    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        lr_img = batch["lr"].to(device)      # (B, 3, H_lr, W_lr)
        ref_img = batch["ref"].to(device)    # (B, 3, H_ref, W_ref)
        gt_flat = batch["gt"].to(device)     # (B, N, 3)
        coords = batch["coords"].to(device)  # (B, N, 2)
        cell = batch["cell"].to(device)      # (B, N, 2)

        # Forward pass
        pred = model(lr_img, ref_img, coords, cell)  # (B, N, 3)

        # Reshape for perceptual loss (may fail for non-square sizes; skip gracefully)
        pred_img, gt_img = None, None
        try:
            B, N, _ = pred.shape
            h = w = int(N ** 0.5)
            if h * w == N:
                pred_img = pred.permute(0, 2, 1).view(B, 3, h, w).clamp(0, 1)
                gt_img = gt_flat.permute(0, 2, 1).view(B, 3, h, w).clamp(0, 1)
        except Exception:
            pass

        losses = criterion(
            pred, gt_flat, pred_img=pred_img, gt_img=gt_img
        )

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in losses.items():
            meters[k].update(v.item(), lr_img.size(0))

        total_steps += 1
        if total_steps % 100 == 0:
            for k, m in meters.items():
                writer.add_scalar(f"train/{k}", m.avg, total_steps)

    log.info(
        f"Epoch {epoch} | "
        + " | ".join(f"{k}: {m.avg:.4f}" for k, m in meters.items())
    )
    return total_steps


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> float:
    model.eval()
    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    for batch in tqdm(loader, desc="Validation", leave=False):
        lr_img = batch["lr"].to(device)
        ref_img = batch["ref"].to(device)
        gt_flat = batch["gt"].to(device)
        coords = batch["coords"].to(device)
        cell = batch["cell"].to(device)

        pred = model(lr_img, ref_img, coords, cell).clamp(0, 1)

        B, N, C = pred.shape
        try:
            h = w = int(N ** 0.5)
            if h * w == N:
                pred_img = pred.permute(0, 2, 1).view(B, C, h, w)
                gt_img = gt_flat.permute(0, 2, 1).view(B, C, h, w)
                psnr_val = compute_psnr(pred_img, gt_img)
                ssim_val = compute_ssim(pred_img, gt_img)
                psnr_meter.update(psnr_val, B)
                ssim_meter.update(ssim_val, B)
        except Exception:
            pass

    log.info(f"Val Epoch {epoch} | PSNR: {psnr_meter.avg:.2f} | SSIM: {ssim_meter.avg:.4f}")
    writer.add_scalar("val/PSNR", psnr_meter.avg, epoch)
    writer.add_scalar("val/SSIM", ssim_meter.avg, epoch)
    return psnr_meter.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RefGuidedINR")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Build model
    model = RefGuidedINR.from_config(cfg).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader = get_data_loader(cfg, split="train")
    val_loader = get_data_loader(cfg, split="val")

    # Optimisation
    total_epochs = cfg.get("train", {}).get("epochs", 100)
    num_steps = total_epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_steps)
    criterion = CombinedLoss.from_config(cfg).to(device)

    # TensorBoard
    log_dir = cfg.get("train", {}).get("log_dir", "runs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    save_dir = cfg.get("train", {}).get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Resume
    start_epoch = 1
    best_psnr = 0.0
    resume = cfg.get("train", {}).get("resume")
    if resume:
        ckpt = load_checkpoint(resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        log.info(f"Resumed from epoch {start_epoch - 1}")

    save_every = cfg.get("train", {}).get("save_every", 10)
    val_every = cfg.get("train", {}).get("val_every", 5)
    total_steps = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, total_epochs + 1):
        total_steps = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scheduler, writer, epoch, total_steps
        )

        if epoch % val_every == 0 or epoch == total_epochs:
            psnr = validate(model, val_loader, device, writer, epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_psnr": best_psnr},
                    save_dir, "best.pth"
                )
                log.info(f"  ✓ New best PSNR: {best_psnr:.2f} dB — saved best.pth")

        if epoch % save_every == 0:
            save_checkpoint(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_psnr": best_psnr},
                save_dir, f"epoch_{epoch:04d}.pth"
            )

    writer.close()
    log.info(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
