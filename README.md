# RefGuidedINR

**Reference-Guided Arbitrary-Scale Image Super-Resolution with Implicit Neural Representation**

Built upon [yinboc/liif](https://github.com/yinboc/liif) (ECCV 2021), this framework adds a reference HR
image path to guide the INR-based decoder, enabling high-fidelity texture transfer at arbitrary scales.

---

## Architecture Overview

```
LR Image ──► LIIFBackbone ──────────────────────────────►┐
                                                         FeatureFusion ──► INRDecoder ──► HR Pixels
RefHR Image ──► ReferenceEncoder ──► DeformableWarpNet ──►┘
                                         (aligns ref feats to LR domain)
```

### Modules

| Module | File | Description |
|---|---|---|
| `LIIFBackbone` | `models/liif_backbone.py` | Residual encoder for LR query image |
| `ReferenceEncoder` | `models/reference_encoder.py` | HR reference image encoder |
| `DeformableWarpNet` | `models/warping.py` | Learned offset-based feature warping |
| `FeatureFusion` | `models/fusion.py` | Concat or cross-attention fusion |
| `INRDecoder` | `models/inr_decoder.py` | Coordinate-conditioned MLP (LIIF-style) |
| `RefGuidedINR` | `models/refguidedinr.py` | Full integrated model |

---

## Installation

```bash
git clone https://github.com/jier6975-droid/RefGuidedINR.git
cd RefGuidedINR
pip install -r requirements.txt
python setup.py develop
```

---

## Data Format

Populate `datasets/` with image triplets that share identical filenames:

```
datasets/
    train/
        LR/        ← low-resolution query images   (e.g., 0001.png)
        RefHR/     ← reference HR images            (e.g., 0001.png)
        GT/        ← ground-truth HR images         (e.g., 0001.png)
    val/
        LR/
        RefHR/
        GT/
```

---

## Quick Start

### Training

```bash
# Using the bash wrapper (recommended)
bash scripts/run_train.sh configs/example_config.yaml 0

# Or directly
python trainers/train.py --config configs/example_config.yaml --gpu 0
```

### Evaluation

```bash
# Using the bash wrapper
bash scripts/run_eval.sh configs/example_config.yaml checkpoints/best.pth 0

# Or directly (optionally save predicted images with --save_images)
python evaluators/evaluate.py \
    --config configs/example_config.yaml \
    --checkpoint checkpoints/best.pth \
    --save_images \
    --gpu 0
```

---

## Configuration

Edit `configs/example_config.yaml` to configure model, data, and training hyperparameters:

```yaml
model:
  backbone_channels: 64
  warper: "deformable"   # currently only 'deformable'
  fusion: "concat"       # 'concat' or 'attention'
  mlp:
    hidden_dim: 256
    num_layers: 4

train:
  lr: 1.0e-4
  epochs: 100
  loss_weights:
    reconstruction: 1.0
    identity: 0.5
    perceptual: 0.1
```

---

## Project Structure

```
RefGuidedINR/
├── configs/
│   └── example_config.yaml
├── data/
│   ├── __init__.py
│   └── data_loader.py          ← Dataset + DataLoader for (LR, RefHR, GT) triplets
├── datasets/                   ← populate with your images
├── evaluators/
│   └── evaluate.py             ← PSNR / SSIM / LPIPS evaluation
├── models/
│   ├── __init__.py
│   ├── liif_backbone.py
│   ├── reference_encoder.py
│   ├── warping.py
│   ├── fusion.py
│   ├── inr_decoder.py
│   └── refguidedinr.py         ← top-level model
├── pretrained_models/          ← place downloaded weights here
├── scripts/
│   ├── run_train.sh
│   └── run_eval.sh
├── trainers/
│   └── train.py                ← full training loop
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── losses.py               ← reconstruction + identity + perceptual
│   └── metrics.py              ← PSNR, SSIM, LPIPS, AverageMeter
├── requirements.txt
├── setup.py
└── README.md
```

---

## Losses

| Loss | Description |
|---|---|
| **Reconstruction** | L1 loss between predicted and GT pixels |
| **Identity** | L1 loss when using reference as pseudo-GT; encourages texture fidelity |
| **Perceptual** | VGG-16 feature-space L1 loss (relu2_2 + relu3_3) |

---

## Metrics

| Metric | Higher/Lower is better |
|---|---|
| PSNR (dB) | ↑ Higher |
| SSIM | ↑ Higher (max 1.0) |
| LPIPS | ↓ Lower |

---

## References

- Chen, Y. et al. *Learning Continuous Image Representation with Local Implicit Image Function*. CVPR 2021. ([yinboc/liif](https://github.com/yinboc/liif))
- Zhang, R. et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018.

---

## License

MIT
