# Training & Validation Workflow

This guide describes how to reproduce the training and validation pipeline for Ultra‑Fine‑Grained Visual Classification (UFGVC) with **Instance‑Aware Hyperbolic Activation (IHA)** and **Batchless Mutual‑Covariance Normalization (BMCN)** layers.

---

## 1. Environment Setup

| Component | Version / Setting            |
| --------- | ---------------------------- |
| OS        | Ubuntu 22.04 LTS             |
| Python    | 3.10                         |
| CUDA      | 12.2                         |
| PyTorch   | 2.3                          |
| timm      | 1.0.3                        |
| Other     | torchvision 0.18, wandb 0.17 |

Activate a clean virtualenv and install:

```bash
pip install torch==2.3 timm==1.0.3 torchvision==0.18 wandb==0.17
```

---

## 2. Data Preparation

1. **Datasets**: Clone the *UFG Benchmark* repository and download all six sub‑datasets (SoyLocal, SoyGlobal, Cotton80, etc.).
2. **Directory Layout**:

   ```
   data/
     └── ufg/
         ├── SoyLocal/
         ├── SoyGlobal/
         └── ...
   ```
3. **Splits**: Use the official train/val CSV splits. If absent, randomly stratify 80 % train / 20 % val using identical seeds for all experiments.

---

## 3. Data Augmentation

| Phase | Augmentations (Compose order)                                                                                             | Hyper‑params |
| ----- | ------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Train | RandomResizedCrop 224, HorizontalFlip p = 0.5, ColorJitter (0.2/0.2/0.2/0.1), RandAugment N=2 M=9, RandomErasing p = 0.25 | default      |
| Val   | Resize 256 → CenterCrop 224                                                                                               | –            |

All datasets share identical transform chains to keep comparisons fair.

---

## 4. Model Instantiation

```python
import timm
from iha_layer import iha_act_layer
from bmcn_layer import bmcn_norm_layer

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    act_layer=iha_act_layer(kappa_init=0.1, per_channel=True),
    norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1, affine=True)
)
```

* **Classifier Head**: Replace final `nn.Linear` with `nn.Linear(embed_dim, num_classes, bias=True)` per dataset.

---

## 5. Optimizer & Scheduler

| Component         | Value                                  |
| ----------------- | -------------------------------------- |
| Optimizer         | AdamW (β₁ = 0.9, β₂ = 0.999)           |
| LR Base           | 5 × 10⁻⁴                               |
| Weight Decay      | 0.05                                   |
| Scheduler         | Cosine w/ linear warm‑up 5 % of epochs |
| Gradient Clipping | 5.0 (global ℓ₂‑norm)                   |

Learning‑rate is automatically scaled by **batch\_size / 64** following the training recipe.

---

## 6. Training Loop (Pseudo‑code)

```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss  = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    scheduler.step()

    # -----------------------------------------
    # Validation at the end of each epoch
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device))
            update_metrics(preds, labels.to(device))
```

### Checkpointing & Logging

* **Every epoch** save `model.state_dict()` plus optimizer/scheduler state.
* **W\&B**: Log train/val loss, accuracy, LR, κ statistics, and running means/vars.
* Keep the **three best** checkpoints by highest Val Top‑1.

---

## 7. Mixed Precision & Small‑Batch Strategy

* Enable `torch.cuda.amp.autocast()` to cut memory by \~30 %.
* If batch\_size < 8, accumulate gradients over `k = 8 // batch_size` steps.

---

## 8. Reproducibility

* Fix seeds `torch.manual_seed(42)` and deterministic CuDNN.
* Save `config.yaml` capturing all hyper‑params.

---

## 9. Expected Runtime

| Backbone  | Batch 4 / 1 GPU | Epochs | Total Time |
| --------- | --------------- | ------ | ---------- |
| Swin‑Tiny | \~2 m / epoch   | 50     | \~1.7 h    |
| ResNet‑50 | \~1 m / epoch   | 50     | \~0.9 h    |

---

> **Tip** — During debugging, shorten to 5 epochs with `--fast_dev_run` flag to verify pipeline integrity.
