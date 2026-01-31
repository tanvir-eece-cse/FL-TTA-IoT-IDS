# Lightning AI L40S Quick Training Guide

## ✅ Complete — Ready to Run!

Your entire codebase has been pushed to: **https://github.com/tanvir-eece-cse/FL-TTA-IoT-IDS**

---

## 🚀 Lightning AI Terminal Commands (1/4th Training)

### Step 1: Clone Repository on Lightning AI
```bash
git clone https://github.com/tanvir-eece-cse/FL-TTA-IoT-IDS.git
cd FL-TTA-IoT-IDS
```

### Step 2: One-Command Setup
```bash
python setup_lightning.py
```

**What this does:**
- Installs all dependencies from `requirements.txt`
- Checks GPU (confirms L40S)
- Downloads real IoT-23 test data (2 scenarios, ~10K flows)

**Time:** ~5 minutes

---

### Step 3: Run Quick Training (Choose ONE)

#### Option A: With WandB Logging (Recommended)
```bash
python train_lightning_quick.py --wandb-key YOUR_WANDB_API_KEY
```

**Get your WandB key from:** https://wandb.ai/authorize

The script will also prompt for your WandB username if not provided:
```bash
python train_lightning_quick.py
# It will ask for: WandB API key and username
```

#### Option B: Without WandB (Faster, No Dashboard)
```bash
python train_lightning_quick.py --skip-wandb
```

---

## ⏱️ Time & Cost Estimates

| Stage | Time | Cost (L40S @ $1.79/hr) |
|-------|------|------------------------|
| Setup | 5 min | ~$0.15 |
| Quick Training (1/4th) | 30-45 min | $0.90-$1.35 |
| **Total** | **35-50 min** | **$1.05-$1.50** |

Full training (if needed): 8-10 hours, ~$15-20

---

## 📊 What You'll Get

1. **Training Metrics**
   - Per-round validation F1, accuracy, loss
   - Per-client performance breakdown
   - Drift detection statistics

2. **Final Results**
   - Test accuracy and F1 scores
   - Confusion matrices per attack type
   - TTA adaptation statistics

3. **Saved Outputs** (in `experiments_quick/`)
   - Trained model weights (`final_model.pt`)
   - Training history JSON
   - Per-client evaluation results

4. **WandB Dashboard** (if enabled)
   - Real-time training curves
   - System metrics (GPU utilization, memory)
   - Experiment comparison

---

## 🔧 Configuration Details

The quick training uses **`configs/l40s_quick.yaml`**:

- **Data:** 25% of full dataset (sampled randomly)
- **Epochs:** 15 (vs 100 full)
- **FL Rounds:** 20 (vs 80 full)
- **Batch Size:** 4096 (optimal for L40S)
- **Model:** MLP with [512, 256, 128] hidden layers
- **TTA:** Drift-aware with entropy threshold 0.28

---

## 📁 Output Structure

```
FL-TTA-IoT-IDS/
├── data/
│   └── processed/
│       └── iot23_real_sampled_25pct/  # Auto-generated 1/4 data
├── experiments_quick/
│   └── federated_tta_YYYYMMDD_HHMMSS/
│       ├── final_model.pt
│       ├── history.json
│       └── tta_results/
└── wandb/  # WandB logs (if enabled)
```

---

## 🎯 Expected Results (Quick Training)

Based on 2-scenario IoT-23 validation:

| Metric | Quick (1/4th) | Full Training |
|--------|---------------|---------------|
| Val F1 | ~97-98% | 99.57% |
| Accuracy | ~98-99% | 99.81% |
| Adaptation Rate | 15-25% | 10-20% |

**Note:** Quick training trades ~1-2% accuracy for 20x speedup.

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA L40S`

### Data Download Fails
```bash
# Manual download
python test_real_iot23.py
```

### WandB Login Issues
```bash
# Alternative: set environment variable
export WANDB_API_KEY=your_key_here
python train_lightning_quick.py --skip-wandb  # then remove --skip-wandb
```

### Out of Memory
```bash
# Reduce batch size in config
# Edit configs/l40s_quick.yaml:
batch_size: 2048  # from 4096
```

---

## 🔄 Full Training (If Needed Later)

```bash
# Download full small subset
python scripts/download_iot23.py --subset small

# Preprocess all scenarios
python scripts/preprocess.py

# Full training
python train.py --mode federated_tta --config configs/l40s_full.yaml --tta drift_aware
```

---

## 📝 Citation & Paper

This codebase implements the methodology from:

**"Drift-Aware Federated Test-Time Adaptation for IoT Intrusion Detection on Real-World IoT-23 Traffic"**

LaTeX methodology diagrams: `paper/methodology_flow_diagrams.tex`

Compile with XeLaTeX for publication-ready figures.

---

## 🆘 Support

GitHub Issues: https://github.com/tanvir-eece-cse/FL-TTA-IoT-IDS/issues

For quick questions, check:
- `README.md` - Full documentation
- `configs/l40s_quick.yaml` - Training config
- `train_lightning_quick.py` - Main script source
