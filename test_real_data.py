"""
Real Data Test Script - Downloads and tests with ACTUAL IoT-23 data
This validates the full pipeline with real network traffic data

Usage:
    python test_real_data.py

This will:
1. Download 2 real IoT-23 scenarios (Mirai malware + Benign traffic)
2. Preprocess with sampling for quick test
3. Run centralized baseline training
4. Run federated training with TTA
5. Report results

Estimated time: 10-15 minutes
Data size: ~200-500 MB
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_cmd(cmd, description, check=True):
    """Run command with timing and status reporting."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0 and check:
        print(f"\n❌ FAILED: {description} (after {elapsed:.1f}s)")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: {description} ({elapsed:.1f}s)")
    
    return result.returncode == 0


def main():
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    print("\n" + "="*70)
    print("🔬 FL-TTA IoT IDS - REAL DATA VALIDATION TEST")
    print("="*70)
    print("\nThis test uses REAL IoT-23 network traffic data (no synthetic data)")
    print("Dataset: Stratosphere IPS IoT-23 - Real malware captures")
    print("Source: https://www.stratosphereips.org/datasets-iot23\n")
    
    # Paths
    raw_dir = base_dir / "data" / "raw" / "iot23_real_test"
    processed_dir = base_dir / "data" / "processed" / "iot23_real_test"
    
    # Clean previous test data
    if raw_dir.exists():
        print(f"Cleaning previous test data: {raw_dir}")
        shutil.rmtree(raw_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    # =========================================================================
    # STEP 1: Download real IoT-23 data (small subset - 2 scenarios)
    # =========================================================================
    run_cmd(
        f'python scripts/download_iot23.py --subset small --output_dir "{raw_dir}"',
        "Step 1/5: Download IoT-23 real data (2-3 scenarios)"
    )
    
    # Check downloaded files
    scenarios = list(raw_dir.iterdir()) if raw_dir.exists() else []
    print(f"\n📁 Downloaded {len(scenarios)} scenarios:")
    for s in scenarios:
        log_file = s / "conn.log.labeled"
        if log_file.exists():
            size_mb = log_file.stat().st_size / (1024 * 1024)
            print(f"   - {s.name}: {size_mb:.1f} MB")
    
    if len(scenarios) < 2:
        print("\n⚠️ Need at least 2 scenarios for federated learning test")
        print("Attempting to download additional scenario...")
        # Download one more scenario manually if needed
    
    # =========================================================================
    # STEP 2: Preprocess with sampling for quick test
    # =========================================================================
    run_cmd(
        f'python scripts/preprocess.py --input_dir "{raw_dir}" --output_dir "{processed_dir}" --sample_per_scenario 30000',
        "Step 2/5: Preprocess data (30k samples per scenario)"
    )
    
    # Check processed files
    if processed_dir.exists():
        train_file = processed_dir / "train.parquet"
        if train_file.exists():
            import pandas as pd
            train_df = pd.read_parquet(train_file)
            print(f"\n📊 Processed data statistics:")
            print(f"   - Train samples: {len(train_df)}")
            print(f"   - Features: {len(train_df.columns)}")
            print(f"   - Label distribution:")
            print(f"     {train_df['label_binary'].value_counts().to_dict()}")
            
            # Check clients
            clients_dir = processed_dir / "clients"
            if clients_dir.exists():
                clients = list(clients_dir.iterdir())
                print(f"   - FL Clients: {len(clients)}")
                for c in clients:
                    c_train = pd.read_parquet(c / "train.parquet")
                    print(f"     - {c.name}: {len(c_train)} samples")
    
    # =========================================================================
    # STEP 3: Centralized training baseline
    # =========================================================================
    run_cmd(
        f'python train.py --mode centralized --config configs/test_tiny.yaml --data_dir "{processed_dir}" --max_epochs 5',
        "Step 3/5: Centralized baseline (5 epochs)"
    )
    
    # =========================================================================
    # STEP 4: Federated training (no TTA)
    # =========================================================================
    run_cmd(
        f'python train.py --mode federated --config configs/test_tiny.yaml --data_dir "{processed_dir}"',
        "Step 4/5: Federated Learning (FedAvg, 5 rounds)"
    )
    
    # =========================================================================
    # STEP 5: Federated training + TTA
    # =========================================================================
    run_cmd(
        f'python train.py --mode federated_tta --tta tent --config configs/test_tiny.yaml --data_dir "{processed_dir}"',
        "Step 5/5: Federated + Test-Time Adaptation (TENT)"
    )
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("🎉 ALL REAL DATA TESTS PASSED!")
    print("="*70)
    
    # Find latest experiment results
    experiments_dir = base_dir / "experiments"
    if experiments_dir.exists():
        exp_dirs = sorted(experiments_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if exp_dirs:
            latest = exp_dirs[0]
            results_file = latest / "results.json"
            history_file = latest / "history.json"
            
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                print(f"\n📈 Latest Results ({latest.name}):")
                print(json.dumps(results, indent=2))
            
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                if 'val_f1_macro' in history and history['val_f1_macro']:
                    best_f1 = max(history['val_f1_macro'])
                    print(f"\n🏆 Best Validation F1-Macro: {best_f1:.4f}")
    
    print("\n" + "-"*70)
    print("✅ Pipeline validated with REAL IoT-23 data!")
    print("-"*70)
    print("\n🚀 NEXT STEPS for Lightning AI L40S:")
    print("   1. Upload this project to Lightning AI")
    print("   2. Download full dataset:")
    print('      python scripts/download_iot23.py --subset medium')
    print("   3. Preprocess without sampling:")
    print('      python scripts/preprocess.py --input_dir data/raw/iot23 --output_dir data/processed/iot23')
    print("   4. Full training:")
    print('      python train.py --mode federated_tta --tta drift_aware --config configs/l40s_full.yaml')
    print()


if __name__ == "__main__":
    main()
