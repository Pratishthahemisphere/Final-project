# Venomous vs Non-Venomous Snake Detection using an Attention Network

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg)](https://streamlit.io/)

A production-grade, attention-augmented deep learning system for classifying venomous and non-venomous snakes from RGB images. The architecture integrates a torchvision ResNet backbone with a custom **Convolutional Block Attention Module (CBAM)**, clinical uncertainty gating, and a scratch **Grad-CAM** interpretability engine.

---

> ⚠️ **CRITICAL MEDICAL & FIELD SAFETY NOTICE**  
> **This deep learning system is an educational and decision-support demonstration.** It must **NEVER** be used to decide whether to approach, handle, or treat a snake encounter. In all field and envenomation scenarios, assume any unidentified snake is dangerous and immediately contact qualified wildlife professionals or emergency medical services.

---

## 🏗️ Architecture Overview

The network augments a deep residual network with CBAM attention after each of the 4 residual stages:

```text
Input Image [B, 3, 224, 224]
        │
   [Stem: 7x7 Conv + BN + ReLU + MaxPool]
        │
   [Stage 1: ResNet Blocks] ──► [CBAM Attention 1 (256 ch)]
        │
   [Stage 2: ResNet Blocks] ──► [CBAM Attention 2 (512 ch)]
        │
   [Stage 3: ResNet Blocks] ──► [CBAM Attention 3 (1024 ch)]
        │
   [Stage 4: ResNet Blocks] ──► [CBAM Attention 4 (2048 ch)] ──► [Target Layer for Grad-CAM]
        │
   [Global Average Pooling]
        │
   [Dropout (p=0.3)]
        │
   [Linear Head: 2 Classes (non_venomous=0, venomous=1)]
```

### CBAM Attention Module
1. **Channel Attention**: Interrogates *what* features are meaningful using a shared Multi-Layer Perceptron (MLP) over Global Average Pooling and Global Max Pooling with reduction ratio $r=16$.
   $$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right)$$
2. **Spatial Attention**: Interrogates *where* key morphological patterns reside by computing channel-wise average and max projections followed by a $7 \times 7$ convolution.
   $$\mathbf{M}_s(\mathbf{F}) = \sigma\left(f^{7\times 7}\left([\text{AvgPool}(\mathbf{F}); \text{MaxPool}(\mathbf{F})]\right)\right)$$

---

## 📁 Repository Structure

```text
snake_attention/
├── app/
│   └── streamlit_app.py           # Interactive web UI with Grad-CAM visualization
├── configs/
│   ├── config.yaml                # Master training & model hyperparameters
│   └── species_map.yaml           # Fine-grained species mapping dictionary
├── data/
│   ├── raw/
│   │   ├── venomous/              # Real venomous snake images (.gitkeep)
│   │   └── non_venomous/          # Real non-venomous snake images (.gitkeep)
│   └── processed/                 # Deterministic stratified splits (.gitkeep)
├── outputs/
│   ├── checkpoints/               # Saved model weights (best.pt, last.pt)
│   ├── plots/                     # Confusion matrix, ROC, PR, training curves
│   └── logs/                      # Training history, CSVs, evaluation logs
├── scripts/
│   ├── check_dataset.py           # Pre-flight data integrity & duplicate checker
│   ├── make_smoke_data.py         # Synthetic smoke data generator for pipeline testing
│   └── build_binary_from_species.py # Species-to-binary dataset converter
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass configuration loader & validator
│   ├── dataset.py                 # SnakeDataset, MD5 deduplication, and DataLoaders
│   ├── evaluate.py                # Test set evaluation, ROC-AUC, threshold sweep
│   ├── explain.py                 # Scratch Grad-CAM and CBAM visualizer
│   ├── model.py                   # SnakeAttentionNet, CBAM modules, backbone loader
│   ├── predict.py                 # Production inference engine with uncertainty gating
│   ├── train.py                   # Two-phase fine-tuning pipeline with AMP & CosineAnnealing
│   └── utils.py                   # Global seeding, device auto-detect, logging
├── tests/                         # Full Pytest test suite (39 tests)
├── requirements.txt               # Pinned package dependencies
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or 3.11
- Operating System: Windows, macOS, or Linux

### 2. Environment Setup
```bash
git clone https://github.com/Pratishthahemisphere/Final-project.git
cd Final-project

python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify Installation
```bash
pytest -v
```

---

## 🔬 Pipeline Workflow

### 1. Prepare & Validate Data
Place your snake images into:
- `data/raw/venomous/`
- `data/raw/non_venomous/`

Run the pre-flight dataset check to detect corrupted images and cross-class duplicates:
```bash
python scripts/check_dataset.py --data_dir data/raw
```

### 2. Model Training
Train the network using two-phase fine-tuning (backbone frozen during warm-up phase, followed by differential learning rates across backbone and attention heads):
```bash
python -m src.train --config configs/config.yaml
```

To resume training from an interrupted checkpoint:
```bash
python -m src.train --config configs/config.yaml --resume outputs/checkpoints/last.pt
```

### 3. Model Evaluation
Evaluate the trained model on the held-out test split:
```bash
python -m src.evaluate --config configs/config.yaml
```
Outputs saved to `outputs/plots/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `misclassified_examples.png`
- `outputs/logs/threshold_analysis.csv`

### 4. Single-Image & Batch Inference
```bash
# Single image inference:
python -m src.predict --image path/to/snake.jpg

# Batch folder inference:
python -m src.predict --image path/to/images/ --threshold 0.70
```

### 5. Explainability (Grad-CAM & CBAM Spatial Attention)
Generate 3-panel visual explanations:
```bash
python -m src.explain --image path/to/snake.jpg --checkpoint outputs/checkpoints/best.pt --out outputs/plots/explain.png
```

### 6. Interactive Web App
Launch the Streamlit web dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Safety Metrics & Uncertainty Gating

In clinical and safety-critical domains, standard accuracy is insufficient. This project prioritizes:
- **Venomous Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — ensures true venomous snakes are correctly identified.
- **False-Safe Rate:** $\frac{FN}{TP + FN}$ — measures the dangerous failure mode where a venomous snake is mistakenly flagged as safe.
- **Clinical Uncertainty Gating:** Predictions where the maximum softmax probability falls below the confidence cutoff (default $0.70$) are flagged as `UNCERTAIN` and deferred to human expert review.

---

## 📜 License
This project is licensed under the MIT License.
