# Experiments & Data Collection Plan

This document outlines the full experimental matrix and the metrics that must be logged to validate the effectiveness of **IHA** and **BMCN** on Ultra‑Fine‑Grained Visual Classification (UFGVC).

---

## 1. Baselines & Variants

| ID     | Activation              | Normalisation         | Notes                                    |
| ------ | ----------------------- | --------------------- | ---------------------------------------- |
| **B0** | GELU                    | LayerNorm             | Default Swin‑T (timm)                    |
| **B1** | ReLU                    | BatchNorm             | Default ResNet‑50                        |
| **P1** | **IHA (shared κ)**      | BatchNorm             | Tests only activation change             |
| **P2** | **IHA (per‑channel κ)** | BatchNorm             | Ablation on κ granularity                |
| **P3** | GELU                    | **BMCN**              | Tests only normalisation change          |
| **P4** | **IHA (shared κ)**      | **BMCN**              | Full proposal, shared κ                  |
| **P5** | **IHA (per‑channel κ)** | **BMCN**              | Full proposal, per‑channel κ             |
| **A1** | IHA κ = 0.0             | BMCN                  | Degenerates to linear act (sanity check) |
| **A2** | IHA κ learnable         | BMCN (momentum = 0.0) | BMCN without running stats               |
| **A3** | IHA (per‑channel)       | BMCN (affine=False)   | Tests γ/β importance                     |

All models are trained with identical data splits, augmentations, and optimiser schedules (see *Training & Validation Workflow*).

---

## 2. Hyper‑parameter Sweep

| Parameter  | Range                 | Purpose                                           |
| ---------- | --------------------- | ------------------------------------------------- |
| κ\_init    | {0.05, 0.1, 0.2, 0.5} | Sensitivity of curvature init                     |
| Momentum   | {0.05, 0.1, 0.2}      | Stability vs responsiveness of BMCN running stats |
| Batch Size | {4, 8, 16}            | Robustness to tiny batches                        |

Use W\&B sweeps (Bayes) with max\_runs = 24.

---

## 3. Evaluation Metrics

| Category              | Metric                                    | Description                    |
| --------------------- | ----------------------------------------- | ------------------------------ |
| **Accuracy**          | **Top‑1 / Top‑5**                         | Overall classification success |
| **Fine‑Grained**      | Class‑wise F1, Macro‑Recall               | Highlights minority classes    |
| **Embedding Quality** | NMI & ARI on feature space                | Post‑hoc clustering quality    |
| **Margin Analysis**   | Inter‑/Intra‑class cosine distance        | Confirms feature separation    |
| **Calibration**       | ECE (Expected Calibration Error)          | Reliability of softmax outputs |
| **Efficiency**        | Params, FLOPs, Peak GPU Mem               | Cost comparison                |
| **Training Dyn.**     | Loss & κ trajectory, Running μ/σ variance | Convergence diagnostics        |

All metrics are computed on the official **val** split. In addition, log **per‑epoch** train loss/acc to analyse over‑fitting.

---

## 4. Data Logging & Storage

1. **W\&B Runs**: Each experiment → one run; store config, metrics, and system resources.
2. **Checkpoints**: Save every 5 epochs & best; keep only final + best three on disk, upload to W\&B Artifacts.
3. **Feature Dumps**: After training, forward entire val split, save penultimate‑layer embeddings (`.npy`) for further analysis.
4. **Confusion Matrices**: Serialize as PNG + CSV each run.
5. **Grad Norms**: Log `torch.nn.utils.clip_grad_norm_` pre‑clip values every 100 steps.

---

## 5. Statistical Testing

* After all runs, perform **paired t‑test** between B0 vs P5 Top‑1 across 6 datasets.
* Report p‑values and 95 % confidence intervals.

---

## 6. Expected Outcomes

| Hypothesis                   | Success Criterion                        |
| ---------------------------- | ---------------------------------------- |
| IHA increases class margin   | ≥ +1.5 pp Top‑1 vs B0 on 4/6 datasets    |
| BMCN stabilises tiny batches | Var(loss\_over\_epochs) ≤ B0 var by 30 % |
| Full stack reduces FLOPs     | ΔFLOPs ≤ +2 % over baseline              |

---

<!-- ## 7. Reproducibility Checklist -->

<!-- * [x] Fixed seeds & deterministic CuDNN. -->
<!-- * [x] Log git commit hash + `pip freeze`. -->
<!-- * [x] Upload all YAML configs & scripts alongside results. -->
<!-- * [x] Provide Colab notebook for sanity‑check run on Cotton80. -->

---

> **Deliverables**: A zipped W\&B export, checkpoint weights, and a `results.xlsx` summarising all metrics.
