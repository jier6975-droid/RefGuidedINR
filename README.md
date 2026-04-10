# RefGuidedINR

Reference-Guided Arbitrary-Scale Image Super-Resolution with Implicit Neural Representation (INR), based on [yinboc/liif](https://github.com/yinboc/liif).

## Features

- Query LR and reference HR encoders
- Learnable feature warping for spatial alignment
- Feature fusion (concat/cross-attention)
- INR-based MLP decoder for arbitrary coordinate queries
- Data loading for (LR, Reference HR, GT HR) triplets
- Configurable training and evaluation
- Modular, extensible codebase

## Installation

```bash
pip install -r requirements.txt
python setup.py develop
```

## Data Format

Place your datasets in `datasets/`, structure:
```
datasets/
    train/
        LR/
        RefHR/
        GT/
    val/
        LR/
        RefHR/
        GT/
```

## Quick Start

**Training:**
```bash
python trainers/train.py --config configs/example_config.yaml
```

**Evaluation:**
```bash
python evaluators/evaluate.py --config configs/example_config.yaml
```

## Reference

- [yinboc/liif](https://github.com/yinboc/liif) (ECCV 2020)
- Your citation here
