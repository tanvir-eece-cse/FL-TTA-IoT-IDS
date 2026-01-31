# FL-TTA-IoT-IDS: Drift-Aware Federated Test-Time Adaptation for IoT Intrusion Detection

## Research Paper: IEEE Local Conference Submission

### Novel Contributions
1. **Real Federated Evaluation**: Non-IID client splits using IoT-23 scenarios as realistic "devices"
2. **Drift-Aware TTA**: Lightweight test-time adaptation (TENT/EATA) for concept drift robustness
3. **Time-Ordered Splits**: Realistic temporal evaluation (no random splits that inflate accuracy)
4. **Reproducible Pipeline**: Full code, configs, and preprocessing for IoT-23 real dataset

### Target Metrics (vs Reference Paper's 94.23% accuracy)
- **Macro-F1 > 92%** (more meaningful than accuracy for imbalanced data)
- **Accuracy > 95%** under realistic non-IID + drift conditions
- **FPR < 5%** (critical for IDS deployment)

### Hardware Requirements
- Lightning AI L40S GPU (48GB VRAM)
- ~8-10 hours training time (fits 15 credits @ 1.79/hr)

## Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download IoT-23 dataset (small labeled subset ~1.2GB)
python scripts/download_iot23.py --subset small

# 3. Preprocess and create client splits
python scripts/preprocess.py --config configs/iot23_federated.yaml

# 4. Test with tiny subset (verification)
python train.py --config configs/test_tiny.yaml --max_epochs 2

# 5. Full training on L40S
python train.py --config configs/l40s_full.yaml
```

## Project Structure
```
FL-TTA-IoT-IDS/
├── configs/           # Hydra configs for experiments
├── data/              # Dataset storage (gitignored)
├── scripts/           # Download & preprocessing
├── src/
│   ├── data/          # DataModules and loaders
│   ├── models/        # MLP, TabTransformer, FT-Transformer
│   ├── fl/            # Federated Learning (Flower)
│   ├── tta/           # Test-Time Adaptation (TENT/EATA)
│   └── utils/         # Metrics, logging, etc.
├── experiments/       # Experiment outputs
├── paper/             # LaTeX paper draft
└── notebooks/         # Analysis notebooks
```

## Datasets
- **IoT-23** (Primary): Real malware traffic captures, 23 scenarios as clients
- **Edge-IIoTset** (Secondary): Cross-dataset generalization test

## Research Gap Addressed
Reference paper (Basry et al. / IJECI 2024) achieved 94.23% but:
- No real federated setup (just architecture description)
- Unknown Kaggle dataset, not reproducible
- Random splits (unrealistic), no drift evaluation
- Binary only, no attack-type breakdown

Our improvements:
- Real FL with Flower, Dirichlet non-IID splits
- IoT-23 with full provenance and preprocessing
- Time-ordered + scenario-based splits for drift
- TTA for deployment robustness without retraining
