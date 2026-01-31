"""
Run Complete Experiment Suite for IEEE Paper
Executes all experiments in sequence with proper logging

This script runs:
1. Centralized baseline (no FL, no TTA)
2. FedAvg baseline (FL without TTA)  
3. FedAvg + TENT (FL with entropy minimization TTA)
4. FedAvg + EATA (FL with efficient sample-aware TTA)
5. FedAvg + Drift-Aware TTA (FL with drift detection + TTA)

Total estimated time on L40S: ~6-8 hours
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time


def run_experiment(name: str, mode: str, tta: str = None, config: str = "configs/l40s_full.yaml"):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"Mode: {mode}, TTA: {tta}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "train.py",
        "--mode", mode,
        "--config", config,
        "--output_dir", f"experiments/{name}"
    ]
    
    if tta:
        cmd.extend(["--tta", tta])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    return {
        "name": name,
        "mode": mode,
        "tta": tta,
        "return_code": result.returncode,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600
    }


def main():
    print("="*60)
    print("FL-TTA IoT IDS - Full Experiment Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check data exists
    data_dir = Path("data/processed/iot23")
    if not data_dir.exists():
        print("ERROR: Processed data not found!")
        print("Run these commands first:")
        print("  python scripts/download_iot23.py --subset small")
        print("  python scripts/preprocess.py")
        sys.exit(1)
    
    results = []
    total_start = time.time()
    
    # 1. Centralized baseline
    results.append(run_experiment(
        name="01_centralized_baseline",
        mode="centralized",
        config="configs/l40s_full.yaml"
    ))
    
    # 2. FedAvg baseline (no TTA)
    results.append(run_experiment(
        name="02_fedavg_baseline",
        mode="federated",
        config="configs/l40s_full.yaml"
    ))
    
    # 3. FedAvg + TENT
    results.append(run_experiment(
        name="03_fedavg_tent",
        mode="federated_tta",
        tta="tent",
        config="configs/l40s_full.yaml"
    ))
    
    # 4. FedAvg + EATA
    results.append(run_experiment(
        name="04_fedavg_eata",
        mode="federated_tta",
        tta="eata",
        config="configs/l40s_full.yaml"
    ))
    
    # 5. FedAvg + Drift-Aware TTA
    results.append(run_experiment(
        name="05_fedavg_drift_aware",
        mode="federated_tta",
        tta="drift_aware",
        config="configs/l40s_full.yaml"
    ))
    
    total_elapsed = time.time() - total_start
    
    # Save results summary
    summary = {
        "experiments": results,
        "total_elapsed_hours": total_elapsed / 3600,
        "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open("experiments/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*60)
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"Estimated cost: ${total_elapsed/3600 * 1.79:.2f}")
    print("\nResults:")
    for r in results:
        status = "✓" if r["return_code"] == 0 else "✗"
        print(f"  {status} {r['name']}: {r['elapsed_hours']:.2f}h")
    
    print(f"\nResults saved to: experiments/experiment_summary.json")
    print("\nNext: Generate paper figures with:")
    print("  python -m src.analysis.evaluation --experiment_dir experiments/")


if __name__ == "__main__":
    main()
