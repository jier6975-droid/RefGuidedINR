# Transformer Model Implementation

## Overview

This directory contains a complete implementation of the Transformer model from the paper **"Attention Is All You Need"** (Vaswani et al., NIPS 2017).

## Files

- `models/transformer.py` - Complete Transformer architecture implementation
- `scripts/run_transformer_demo.py` - Training and evaluation script
- `transformer_results/` - Training results and outputs
  - `best_model.pth` - Trained model checkpoint (96MB)
  - `training_curves.png` - Training/validation loss and accuracy curves
  - `results.json` - Detailed training metrics
  - `REPORT_CN.md` - Comprehensive Chinese report

## Quick Start

### Requirements

```bash
pip install torch torchvision tqdm matplotlib
```

### Run the Demo

```bash
python scripts/run_transformer_demo.py
```

This will:
1. Create a synthetic sequence copying dataset
2. Train the Transformer model for 5 epochs
3. Evaluate the model and generate examples
4. Save results to `transformer_results/`

## Model Architecture

The implementation includes all core components from the original paper:

- ✅ Multi-Head Attention
- ✅ Positional Encoding (sine/cosine)
- ✅ Position-wise Feed-Forward Networks
- ✅ Encoder Stack (4 layers)
- ✅ Decoder Stack (4 layers)
- ✅ Residual Connections
- ✅ Layer Normalization

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model Dimension (d_model) | 256 |
| Number of Layers | 4 |
| Number of Attention Heads | 8 |
| Feed-Forward Dimension (d_ff) | 1024 |
| Vocabulary Size | 100 |
| **Total Parameters** | **7.45M** |

## Training Results

| Metric | Value |
|--------|-------|
| Training Time | 277.85 seconds (~4.6 minutes) |
| Best Validation Accuracy | **38.96%** |
| Final Training Loss | 2.491 |
| Final Validation Loss | 1.853 |

### Learning Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 3.731 | 15.05% | 2.893 | 20.51% |
| 2 | 3.095 | 20.80% | 2.398 | 26.67% |
| 3 | 2.800 | 23.49% | 2.133 | 31.29% |
| 4 | 2.636 | 24.83% | 1.996 | 34.45% |
| 5 | 2.491 | 26.58% | 1.853 | **38.96%** |

## Task Description

**Sequence Copying Task**: The model learns to copy input sequences to output.

- Input: Random integer sequences of length 10 (values: 3-99)
- Output: Copy of the input sequence
- Training Samples: 10,000
- Validation Samples: 1,000

This simplified task demonstrates the Transformer's ability to learn sequence patterns.

## Key Features

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention

Allows the model to jointly attend to information from different representation subspaces.

### Positional Encoding

Injects position information using sine and cosine functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Code Structure

```
models/
  └── transformer.py          # Full Transformer implementation
      ├── PositionalEncoding
      ├── MultiHeadAttention
      ├── PositionWiseFeedForward
      ├── EncoderLayer
      ├── DecoderLayer
      ├── Encoder
      ├── Decoder
      └── Transformer

scripts/
  └── run_transformer_demo.py # Training script
      ├── SequenceCopyDataset
      ├── train_epoch()
      ├── evaluate()
      ├── generate_examples()
      └── main()

transformer_results/
  ├── best_model.pth          # Trained model
  ├── training_curves.png     # Visualizations
  ├── results.json            # Detailed metrics
  └── REPORT_CN.md            # Chinese report
```

## Customization

To modify the model or training:

1. **Change model size**: Edit configuration in `run_transformer_demo.py`
   ```python
   config = {
       "d_model": 512,      # Increase model dimension
       "num_layers": 6,     # More layers
       "num_epochs": 20,    # Train longer
       ...
   }
   ```

2. **Use different task**: Create a new dataset class in the training script

3. **Adjust learning rate**: Modify the optimizer settings

## Performance Tips

To improve results:

- ✅ Train for more epochs (20-50)
- ✅ Use learning rate warmup and decay
- ✅ Increase training data
- ✅ Use GPU for faster training
- ✅ Tune hyperparameters (dropout, learning rate, etc.)

## Reference

**Paper**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

**arXiv**: https://arxiv.org/abs/1706.03762

**The Annotated Transformer**: http://nlp.seas.harvard.edu/2018/04/03/attention.html

## License

MIT

---

**实现日期 / Implementation Date**: 2026-05-14

**详细报告 / Detailed Report**: See `REPORT_CN.md` for comprehensive analysis in Chinese
