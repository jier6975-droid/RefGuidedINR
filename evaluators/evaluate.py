"""
evaluate.py — RefGuidedINR Evaluation Pipeline.

Usage:
    python evaluators/evaluate.py --config configs/example_config.yaml \\
        [--checkpoint checkpoints/best.pth] [--save_images] [--gpu 0]

Computes PSNR, SSIM, and LPIPS on the validation split and optionally
saves the reconstructed HR images.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging

import torch
from tqdm import tqdm

from models import RefGuidedINR
from data import get_data_loader
from utils import (
    load_config,
    load_checkpoint,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    AverageMeter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a (3, H, W) tensor in [0, 1] as a PNG image."""
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_image(tensor.clamp(0, 1), path)


@torch.no_grad()
def evaluate(cfg: dict, checkpoint_path: str, save_images: bool, device: torch.device) -> dict:
    model = RefGuidedINR.from_config(cfg).to(device)
    ckpt = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint from {checkpoint_path}")

    val_loader = get_data_loader(cfg, split="val")

    psnr_m = AverageMeter("PSNR")
    ssim_m = AverageMeter("SSIM")
    lpips_m = AverageMeter("LPIPS")

    save_dir = cfg.get("eval", {}).get("save_dir", "results")
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        lr_img = batch["lr"].to(device)
        ref_img = batch["ref"].to(device)
        gt_flat = batch["gt"].to(device)
        coords = batch["coords"].to(device)
        cell = batch["cell"].to(device)

        pred = model(lr_img, ref_img, coords, cell).clamp(0, 1)

        B, N, C = pred.shape
        try:
            h = w = int(N ** 0.5)
            if h * w != N:
                continue
            pred_img = pred.permute(0, 2, 1).view(B, C, h, w)
            gt_img = gt_flat.permute(0, 2, 1).view(B, C, h, w)

            psnr_m.update(compute_psnr(pred_img, gt_img), B)
            ssim_m.update(compute_ssim(pred_img, gt_img), B)
            lpips_m.update(compute_lpips(pred_img, gt_img), B)

            if save_images:
                for b in range(B):
                    idx = i * B + b
                    save_image_tensor(pred_img[b], os.path.join(save_dir, f"{idx:05d}_pred.png"))
                    save_image_tensor(gt_img[b], os.path.join(save_dir, f"{idx:05d}_gt.png"))
        except Exception as e:
            log.warning(f"Skipping batch {i}: {e}")
            continue

    results = {
        "PSNR": psnr_m.avg,
        "SSIM": ssim_m.avg,
        "LPIPS": lpips_m.avg,
    }
    log.info("=" * 40)
    for k, v in results.items():
        log.info(f"  {k}: {v:.4f}")
    log.info("=" * 40)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RefGuidedINR")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--save_images", action="store_true", help="Save predicted HR images")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or cfg.get("eval", {}).get("checkpoint", "checkpoints/best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first with: python trainers/train.py --config configs/example_config.yaml"
        )

    evaluate(cfg, ckpt_path, args.save_images, device)


if __name__ == "__main__":
    main()
