"""
Loss functions for RefGuidedINR training.

Three losses are used:
  - ReconstructionLoss : L1 pixel reconstruction loss between predicted and GT.
  - IdentityLoss       : L1 loss computed when the reference IS the GT
                         (encourages the model to copy reference content faithfully).
  - PerceptualLoss     : VGG-based perceptual loss (requires torchvision).
  - CombinedLoss       : Weighted combination of the above three.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss between predicted and GT pixel values."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class IdentityLoss(nn.Module):
    """
    L1 identity loss: when the reference image IS the GT (or a HR version of
    the input), the model output should closely match it.
    Applied the same way as reconstruction loss but can be weighted separately.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.

    Extracts features from VGG16 relu2_2 and relu3_3 layers and computes
    L1 loss in feature space.

    Args:
        layer_ids: Indices in the VGG feature sequence to extract from.
        use_gpu: Move VGG to GPU if available.
    """

    def __init__(self):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            features = vgg.features
            # relu2_2 = index 9, relu3_3 = index 16
            self.slice1 = nn.Sequential(*list(features.children())[:10])
            self.slice2 = nn.Sequential(*list(features.children())[10:17])
            for param in self.parameters():
                param.requires_grad = False
            self._available = True
        except Exception:
            self._available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image (B, 3, H, W), values in [0, 1].
            target: GT image (B, 3, H, W), values in [0, 1].
        """
        if not self._available:
            return torch.tensor(0.0, device=pred.device)

        loss = F.l1_loss(self.slice1(pred), self.slice1(target))
        loss = loss + F.l1_loss(self.slice2(pred), self.slice2(target))
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted sum of reconstruction, identity, and perceptual losses.

    Args:
        w_reconstruction: Weight for reconstruction loss.
        w_identity: Weight for identity loss.
        w_perceptual: Weight for perceptual loss.
    """

    def __init__(
        self,
        w_reconstruction: float = 1.0,
        w_identity: float = 0.5,
        w_perceptual: float = 0.1,
    ):
        super().__init__()
        self.w_rec = w_reconstruction
        self.w_id = w_identity
        self.w_perc = w_perceptual

        self.rec_loss = ReconstructionLoss()
        self.id_loss = IdentityLoss()
        self.perc_loss = PerceptualLoss()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        pred_img: torch.Tensor = None,
        gt_img: torch.Tensor = None,
        ref_img: torch.Tensor = None,
        pred_ref: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred: Predicted pixels (B, N, 3).
            gt: Ground-truth pixels (B, N, 3).
            pred_img: Predicted image reshaped to (B, 3, H, W) for perceptual loss.
            gt_img: GT image reshaped to (B, 3, H, W) for perceptual loss.
            ref_img: Reference HR image (B, 3, H_ref, W_ref) for identity loss.
            pred_ref: Predicted output on reference coordinates (B, N_ref, 3).

        Returns:
            dict with keys 'total', 'reconstruction', 'identity', 'perceptual'.
        """
        rec = self.rec_loss(pred, gt)

        id_val = torch.tensor(0.0, device=pred.device)
        if ref_img is not None and pred_ref is not None:
            ref_flat = ref_img.view(ref_img.shape[0], 3, -1).permute(0, 2, 1)
            id_val = self.id_loss(pred_ref, ref_flat)

        perc = torch.tensor(0.0, device=pred.device)
        if pred_img is not None and gt_img is not None:
            perc = self.perc_loss(pred_img, gt_img)

        total = self.w_rec * rec + self.w_id * id_val + self.w_perc * perc
        return {
            "total": total,
            "reconstruction": rec,
            "identity": id_val,
            "perceptual": perc,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "CombinedLoss":
        weights = cfg.get("train", {}).get("loss_weights", {})
        return cls(
            w_reconstruction=weights.get("reconstruction", 1.0),
            w_identity=weights.get("identity", 0.5),
            w_perceptual=weights.get("perceptual", 0.1),
        )
