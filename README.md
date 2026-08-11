<div align="center">

# 🌍 XRAYEARTH
### Post-Disaster Building Damage Assessment with Siamese Mask R-CNN
**A 10-Variant Ablation Study on Extreme Class Imbalance using the xBD Dataset**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-purple)

*Post-disaster building damage assessment using Siamese Mask R-CNN + Focal Loss on the xBD dataset*

</div>

---

## 📌 Overview

Post-disaster building damage assessment is critical for emergency
response, but automated systems face severe class imbalance in satellite
imagery.

In the xBD dataset, approximately 78.4% of annotated buildings belong to
the no-damage class, while destroyed buildings represent only 7.4%.

XRAYEARTH investigates whether temporal feature fusion and
difficulty-aware loss optimization can improve detection of these rare
but critical destroyed structures.

| Component | Choice |
|---|---|
| Architecture | Siamese Mask R-CNN (ResNet50-FPN) |
| Fusion | Concat + Difference at FPN level |
| Loss | Focal Loss (γ=2.0 + inverse-frequency class weights)) |
| Primary Metric | Macro F1-score |
| Optimization | AMP FP16 + OneCycleLR |

---
## 💡 Why XRAYEARTH?

The primary challenge is not simply detecting buildings after a disaster,
but correctly identifying the minority classes that are most important
for emergency response.

XRAYEARTH addresses this through three complementary mechanisms:

1. **Temporal Feature Fusion**
   Pre- and post-disaster images are processed through a shared-weight
   Siamese ResNet50-FPN backbone.

2. **Focal Loss**
   Training focuses on difficult and minority-class examples instead
   of allowing abundant easy no-damage examples to dominate the gradient.

3. **Controlled Ablation**
   Ten sequential variants evaluate the impact of architectural,
   optimization, regularization, and loss-function decisions.
   
## 🏗️ XRAYEARTH Architecture

XRAYEARTH processes pre- and post-disaster satellite images using a
shared-weight ResNet50-FPN Siamese backbone.

Feature maps are fused using concatenation and element-wise difference,
providing an explicit temporal change signal to the Mask R-CNN detector.

Pre-disaster image ──→ ResNet50-FPN ──┐
                                      ├→ Feature Fusion
Post-disaster image ─→ ResNet50-FPN ──┘
                                         ↓
                              Concat + Difference
                                         ↓
                                      RPN
                                         ↓
                                    ROI Align
                                    ↙       ↘
                              Cls Head     Mask Head
                              Focal Loss   Binary CE
                                    ↓
                            4-Class Damage

## 🗂️ Dataset

**xBD Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)

Place dataset files as:
```
data/
├── pre/       ← pre-disaster images (.png)
├── post/      ← post-disaster images (.png)
└── labels/    ← JSON annotation files
```

---

## ⚙️ Setup

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/xrayearth.git
cd xrayearth
```

### 2. Environment
```bash
conda env create -f environment.yml
conda activate xrayearth
```

### 3. Configure paths
```bash
cp .env.example .env
# Edit .env with your local paths
```

### 4. Login to WandB
```bash
wandb login
```

---

## 🚀 Training

### Smoke test (Machine A — quick check)
```bash
bash scripts/smoke_test.sh
```

### Full training (Machine B)
```bash
bash scripts/train.sh v1    # Baseline
bash scripts/train.sh v10   # Full system
```

### Run all ablation variants
```bash
for v in v1 v2 v3 v4 v5 v6 v7 v8 v9 v10; do
    bash scripts/train.sh $v
done
```

---

## 📊 Ablation Study

| Version | Description |
|---|---|
| V1 | Baseline — single image, CrossEntropy |
| V2 | + Pretrained ResNet50 |
| V3 | + Deeper classifier head |
| V4 | + Data augmentation |
| V5 | + Freeze backbone |
| V6 | + Full fine-tuning |
| V7 | + Dropout |
| V8 | + GroupNorm |
| V9 | + Lighter backbone (ResNet34) |
| V10 | **Full system** — Siamese + Focal Loss + 512×512 |

---

## 📈 Key Results

| Version | No-Damage AP | Minor-Damage AP | Major-Damage AP | Destroyed AP | **Macro F1** |
| :-----: | -----------: | --------------: | --------------: | -----------: | -----------: |
|  **V1** |       0.4092 |          0.3099 |          0.3333 |       0.1845 |   **0.4684** |
|  **V2** |       0.4774 |          0.3775 |          0.3857 |       0.2552 |   **0.5580** |
|  **V3** |       0.4620 |          0.3672 |          0.3943 |       0.2509 |   **0.5479** |
|  **V4** |       0.4468 |          0.3447 |          0.3709 |       0.2274 |   **0.5274** |
|  **V5** |       0.2880 |          0.2010 |          0.2128 |       0.1445 |   **0.3635** |
|  **V6** |       0.4681 |          0.3822 |          0.4076 |       0.2704 |   **0.5558** |
|  **V7** |       0.4817 |          0.3832 |          0.4059 |       0.2652 |   **0.5736** |
|  **V8** |       0.4893 |          0.3937 |          0.4108 |       0.2585 |   **0.5746** |
|  **V9** |       0.5065 |          0.3626 |          0.4018 |       0.2502 |   **0.5854** |
| **V10** |   **0.5633** |      **0.4041** |      **0.4677** |   **0.4344** |   **0.6270** |

XRAYEARTH V10 achieved:

| Metric | V1 Baseline | V10 XRAYEARTH | Improvement |
|---|---:|---:|---:|
| Macro F1 | 0.4684 | **0.6270** | +33.9% |
| mAP@0.50 | 0.3092 | **0.4674** | +51.2% |
| mAP@0.50:0.95 | 0.1399 | **0.2455** | +75.5% |
| Destroyed AP | 0.1845 | **0.4344** | +135.4% |

## ⚠️ Limitations

The current system has several limitations:

- Spectrally similar terrain can still produce false positives.
- Buildings crossing tile boundaries may be duplicated or missed.
- The V10 configuration introduces Siamese fusion, Focal Loss, and
  512×512 tiling simultaneously, so their individual contributions
  cannot be completely isolated.

## 🏗️ Project Structure

```
xrayearth/
├── src/           ← All Python source code
├── configs/       ← YAML configs (base + v1–v10)
├── scripts/       ← Training + export scripts
├── outputs/       ← Checkpoints, logs, predictions
├── data/          ← Dataset (local only, gitignored)
└── notebooks/     ← EDA and visualization
```

---

## 🖥️ Hardware

| Machine | GPU | Role |
|---|---|---|
| Machine A | RTX 3050 | Development + debugging |
| Machine B | RTX 5060 (8GB) | Full training + TensorRT |

---

## 👥 Team

- **Madhusuthanan G**
- **Shri Harsan M**
- **Pharos Sophy Samuel T J**
- **Dr. Sharanya S** — Faculty Advisor

## 📈 Tracking

All experiments logged to **WandB** under project `xrayearth`.  
Compare ablation runs via WandB parallel coordinates on `val/macro_f1`.

---

<div align="center">
Built with ❤️ for disaster response AI
</div>
