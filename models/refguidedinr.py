"""
RefGuidedINR: Top-level model integrating all sub-modules.

Architecture:
  1. LIIFBackbone  → encodes the query LR image into feature maps.
  2. ReferenceEncoder → encodes the reference HR image into feature maps.
  3. DeformableWarpNet → aligns reference features to the query domain.
  4. FeatureFusion → merges query and warped reference features.
  5. INRDecoder → maps fused features + continuous coords to RGB values.
"""
import torch
import torch.nn as nn
from .liif_backbone import LIIFBackbone
from .reference_encoder import ReferenceEncoder
from .warping import DeformableWarpNet
from .fusion import FeatureFusion
from .inr_decoder import INRDecoder


class RefGuidedINR(nn.Module):
    """
    Reference-Guided Arbitrary-Scale Super-Resolution via INR.

    Args:
        backbone_channels: Feature channels for both LR backbone and reference encoder.
        warper: Warping strategy. Currently only 'deformable' is supported.
        fusion: Fusion mode. Either 'concat' or 'attention'.
        mlp_hidden_dim: Hidden dimension of the INR MLP decoder.
        mlp_num_layers: Number of hidden layers in the INR MLP.
        mlp_out_dim: Output channels (3 for RGB).
        num_res_blocks: Residual blocks in LIIFBackbone and ReferenceEncoder.
    """

    def __init__(
        self,
        backbone_channels: int = 64,
        warper: str = "deformable",
        fusion: str = "concat",
        mlp_hidden_dim: int = 256,
        mlp_num_layers: int = 4,
        mlp_out_dim: int = 3,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        C = backbone_channels

        self.lr_encoder = LIIFBackbone(
            in_channels=3,
            feat_channels=C,
            num_res_blocks=num_res_blocks,
        )
        self.ref_encoder = ReferenceEncoder(
            in_channels=3,
            feat_channels=C,
            num_res_blocks=num_res_blocks,
        )

        if warper == "deformable":
            self.warper = DeformableWarpNet(feat_channels=C)
        else:
            raise ValueError(f"Unknown warper: {warper!r}. Use 'deformable'.")

        self.fusion = FeatureFusion(mode=fusion, feat_channels=C)
        self.decoder = INRDecoder(
            feat_channels=C,
            coord_dim=2,
            cell_dim=2,
            out_dim=mlp_out_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
        )

    def encode(
        self,
        lr_img: torch.Tensor,
        ref_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode and fuse LR + reference features.

        Args:
            lr_img: (B, 3, H_lr, W_lr) — query low-resolution image.
            ref_img: (B, 3, H_ref, W_ref) — reference high-resolution image.

        Returns:
            fused_feat: (B, C, H_lr, W_lr) fused feature map.
        """
        lr_feat = self.lr_encoder(lr_img)        # (B, C, H_lr, W_lr)
        ref_feat = self.ref_encoder(ref_img)      # (B, C, H_ref, W_ref)
        warped_ref = self.warper(ref_feat, lr_feat)  # (B, C, H_lr, W_lr)
        fused = self.fusion(lr_feat, warped_ref)   # (B, C, H_lr, W_lr)
        return fused

    def decode(
        self,
        fused_feat: torch.Tensor,
        coords: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode pixel values from fused features at continuous query coordinates.

        Args:
            fused_feat: (B, C, H, W) fused feature map.
            coords: (B, N, 2) query coordinates in [-1, 1].
            cell: (B, N, 2) cell sizes corresponding to each query point.

        Returns:
            pred: (B, N, 3) predicted RGB values.
        """
        return self.decoder(fused_feat, coords, cell)

    def forward(
        self,
        lr_img: torch.Tensor,
        ref_img: torch.Tensor,
        coords: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            lr_img: (B, 3, H_lr, W_lr) query LR image.
            ref_img: (B, 3, H_ref, W_ref) reference HR image.
            coords: (B, N, 2) target coordinate grid, values in [-1, 1].
            cell: (B, N, 2) cell size vectors.

        Returns:
            pred: (B, N, 3) predicted HR pixel values at queried coordinates.
        """
        fused_feat = self.encode(lr_img, ref_img)
        return self.decode(fused_feat, coords, cell)

    @classmethod
    def from_config(cls, cfg: dict) -> "RefGuidedINR":
        """Instantiate from a parsed YAML config dict."""
        m = cfg.get("model", {})
        mlp_cfg = m.get("mlp", {})
        return cls(
            backbone_channels=m.get("backbone_channels", 64),
            warper=m.get("warper", "deformable"),
            fusion=m.get("fusion", "concat"),
            mlp_hidden_dim=mlp_cfg.get("hidden_dim", 256),
            mlp_num_layers=mlp_cfg.get("num_layers", 4),
            mlp_out_dim=mlp_cfg.get("out_dim", 3),
        )
