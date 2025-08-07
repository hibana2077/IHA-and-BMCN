# IHA and BMCN: Instance-Aware Hyperbolic Activation & Batchless Mutual-Covariance Normalization

This repository implements **Instance-Aware Hyperbolic Activation (IHA)** and **Batchless Mutual-Covariance Normalization (BMCN)** layers for addressing overfitting in Ultra-Fine-Grained Visual Classification (UFGVC) tasks.

## Overview

- **IHA**: A learnable hyperbolic activation function `sinh(κ·(x-μ))/(κ·(σ+ε))` that dynamically adjusts non-linearity based on instance contrast difficulty
- **BMCN**: A batchless normalization technique that estimates statistics per instance and maintains global running averages, making it robust to small batch sizes

## Features

- 🔌 **Plug-and-play**: Drop-in replacement for standard activations (ReLU/GELU) and normalization (BatchNorm/LayerNorm)
- 📊 **Comprehensive evaluation**: Implements all metrics from the experimental plan including NMI, ARI, ECE, margin analysis
- 🎛️ **Configurable experiments**: Support for all baseline and variant configurations (B0-B1, P1-P5, A1-A3)
- 📈 **W&B integration**: Full experiment tracking and hyperparameter sweeps
- 🚀 **timm compatibility**: Works with any timm model architecture

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test the implementation
python test_implementation.py
```

### 2. Basic Training

```bash
# Train with default P5 configuration (IHA per-channel + BMCN)
python train.py --config configs/P5.yaml --dataset cotton80

# Quick debug run (5 epochs)
python train.py --config configs/P5.yaml --debug
```

### 3. Run All Experiments

```bash
# Train all variants on a dataset
for variant in B0 B1 P1 P2 P3 P4 P5 A1 A2 A3; do
    python train.py --config configs/${variant}.yaml --dataset cotton80
done
```

## Project Structure

```
IHA-and-BMCN/
├── src/
│   ├── iha.py              # IHA activation implementation
│   ├── bmcn.py             # BMCN normalization implementation
│   └── dataset/
│       └── ufgvc.py        # UFGVC dataset loader
├── utils/
│   ├── config.py           # Configuration management
│   └── metrics.py          # Comprehensive evaluation metrics
├── configs/                # Configuration files for all variants
├── docs/                   # Experimental design documentation
├── scripts/                # Training scripts for different GPUs
├── train.py               # Main training script
├── evaluate.py            # Model evaluation script
├── sweep.py               # Hyperparameter sweep with W&B
└── test_implementation.py  # Quick test script
```

## Experiment Variants

| Variant | Activation | Normalization | Description |
|---------|------------|---------------|-------------|
| **B0** | GELU | LayerNorm | Default Swin-T baseline |
| **B1** | ReLU | BatchNorm | Default ResNet-50 baseline |
| **P1** | IHA (shared κ) | BatchNorm | Test activation only |
| **P2** | IHA (per-channel κ) | BatchNorm | Ablation on κ granularity |
| **P3** | GELU | BMCN | Test normalization only |
| **P4** | IHA (shared κ) | BMCN | Full proposal (shared) |
| **P5** | IHA (per-channel κ) | BMCN | Full proposal (per-channel) |
| **A1** | IHA (κ=0) | BMCN | Sanity check (linear activation) |
| **A2** | IHA | BMCN (momentum=0) | No running statistics |
| **A3** | IHA (per-channel) | BMCN (no affine) | No γ/β parameters |

## Dataset Support

Supports all UFGVC datasets with automatic download:
- `cotton80`: Cotton classification (80 classes)
- `soybean`: Soybean classification  
- `soy_ageing_r1` to `soy_ageing_r6`: Soybean aging datasets

Each dataset includes pre-split train/val/test sets.

## Advanced Usage

### Hyperparameter Sweeps

```bash
# Create W&B sweep
python sweep.py --create

# Run sweep agent (replace SWEEP_ID)
python sweep.py --run --sweep_id SWEEP_ID --count 24
```

### Model Evaluation

```bash
# Evaluate single model
python evaluate.py --checkpoint outputs/model/best_checkpoint.pth --dataset cotton80

# Compare multiple models with statistical testing
python evaluate.py --checkpoints model1.pth model2.pth model3.pth --compare --statistical_test
```

### Custom Configuration

```python
from utils.config import Config

# Create custom config
config = Config()
config.model.iha.kappa_init = 0.2
config.model.iha.per_channel = True
config.training.batch_size = 8
config.experiment.variant = "P5"

# Save and use
config.save_yaml("custom_config.yaml")
```

## Implementation Details

### IHA Activation

```python
from src.iha import iha_act_layer
import timm

# Use with any timm model
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    act_layer=iha_act_layer(kappa_init=0.1, per_channel=True)
)
```

### BMCN Normalization

```python
from src.bmcn import bmcn_norm_layer
import timm

# CNN models
model = timm.create_model(
    'resnet50',
    pretrained=True,
    norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1)
)

# ViT models (channel_last=True)
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    norm_layer=bmcn_norm_layer(channel_last=True)
)
```

## Key Features

### Robust to Small Batches
BMCN works reliably with batch sizes as small as 1, unlike BatchNorm which requires larger batches.

### Adaptive Non-linearity  
IHA learns optimal curvature κ per instance, providing more gradient signal for difficult samples.

### Comprehensive Metrics
Evaluation includes:
- Accuracy (Top-1, Top-5)
- Fine-grained metrics (per-class F1, macro recall)
- Embedding quality (NMI, ARI via clustering)
- Margin analysis (inter/intra-class distances)
- Calibration (ECE)
- Efficiency (parameters, FLOPs, memory)

### Statistical Validation
Built-in paired t-tests for comparing model performance across datasets.

## Configuration

Each experiment variant has a corresponding YAML config in `configs/`. Key parameters:

```yaml
model:
  name: "swin_tiny_patch4_window7_224"
  iha:
    kappa_init: 0.1
    per_channel: true
  bmcn:
    eps: 1e-5
    momentum: 0.1

training:
  epochs: 50
  batch_size: 16
  learning_rate: 5e-4
  weight_decay: 0.05

data:
  dataset_name: "cotton80"
  root: "./data"
```

## Expected Results

Based on the experimental design, key hypotheses:
- IHA should improve class margins (≥+1.5pp Top-1 vs baseline on 4/6 datasets)
- BMCN should stabilize training with small batches (30% reduction in loss variance)
- Combined approach should maintain efficiency (≤+2% FLOPs vs baseline)

## Contributing

1. All experiments follow configurations in `docs/exp.md`
2. Use the provided configuration system for reproducibility
3. Add comprehensive metrics for any new evaluation approaches
4. Test with `test_implementation.py` before submitting changes

## Citation

If you use this implementation, please cite:

```bibtex
@article{iha_bmcn_2024,
  title={Instance-Aware Hyperbolic Activation and Batchless Mutual-Covariance Normalization: Addressing Overfitting in Ultra-Fine-Grained Visual Classification},
  author={Your Name},
  year={2024}
}
```