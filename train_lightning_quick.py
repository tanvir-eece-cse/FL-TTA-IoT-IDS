"""
Quick Training Script for Lightning AI L40S (1/4th data)
=========================================================
Runs a fast validation experiment with:
- 25% of the full dataset
- Fewer epochs/rounds
- WandB logging with API key prompt

Usage on Lightning AI:
    pip install -r requirements.txt
    python train_lightning_quick.py --wandb-key YOUR_KEY

Local testing:
    python train_lightning_quick.py --debug
"""
import os
import sys
import argparse
from pathlib import Path
import getpass

# Set environment before imports
def setup_wandb(api_key=None, entity=None):
    """Setup WandB with API key."""
    if api_key:
        os.environ['WANDB_API_KEY'] = api_key
    elif not os.environ.get('WANDB_API_KEY'):
        print("\n" + "="*60)
        print("WANDB SETUP")
        print("="*60)
        print("Get your API key from: https://wandb.ai/authorize")
        api_key = getpass.getpass("Enter your WandB API key: ")
        os.environ['WANDB_API_KEY'] = api_key
    
    if entity:
        os.environ['WANDB_ENTITY'] = entity

# Parse args early for wandb setup
parser = argparse.ArgumentParser(description="Quick FL-TTA Training on Lightning AI L40S")
parser.add_argument("--wandb-key", type=str, default=None,
                    help="WandB API key (or will prompt)")
parser.add_argument("--wandb-entity", type=str, default=None,
                    help="WandB username/entity (or will prompt)")
parser.add_argument("--data-dir", type=str, default="data/processed/iot23_real",
                    help="Data directory")
parser.add_argument("--mode", type=str, default="federated_tta",
                    choices=["centralized", "federated", "federated_tta"],
                    help="Training mode")
parser.add_argument("--debug", action="store_true",
                    help="Debug mode (no WandB, minimal data)")
parser.add_argument("--skip-wandb", action="store_true",
                    help="Skip WandB logging")
args = parser.parse_args()

# Setup WandB
if not args.debug and not args.skip_wandb:
    entity = args.wandb_entity
    if not entity:
        print("\nEnter your WandB username (found at https://wandb.ai/settings):")
        entity = input("WandB username: ").strip()
    
    setup_wandb(args.wandb_key, entity)

# Now import everything else
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import json
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from train import (
    setup_l40s_optimizations,
    train_centralized,
    train_federated,
    load_config
)

def sample_data(data_dir: Path, fraction: float = 0.25, seed: int = 42):
    """Sample a fraction of data for quick training."""
    print(f"\n{'='*60}")
    print(f"SAMPLING {fraction*100:.0f}% OF DATA FOR QUICK TRAINING")
    print(f"{'='*60}\n")
    
    data_dir = Path(data_dir)
    sampled_dir = data_dir.parent / f"{data_dir.name}_sampled_{int(fraction*100)}pct"
    sampled_dir.mkdir(exist_ok=True)
    
    # Copy feature columns
    import shutil
    shutil.copy(data_dir / "feature_columns.txt", sampled_dir / "feature_columns.txt")
    
    # Sample main splits
    for split in ['train', 'val', 'test']:
        original = data_dir / f"{split}.parquet"
        if original.exists():
            df = pd.read_parquet(original)
            sampled = df.sample(frac=fraction, random_state=seed)
            sampled.to_parquet(sampled_dir / f"{split}.parquet")
            print(f"  {split}: {len(df)} → {len(sampled)} ({len(sampled)/len(df)*100:.1f}%)")
    
    # Sample client splits
    clients_dir = data_dir / "clients"
    if clients_dir.exists():
        sampled_clients = sampled_dir / "clients"
        sampled_clients.mkdir(exist_ok=True)
        
        for client_dir in clients_dir.iterdir():
            if client_dir.is_dir():
                client_out = sampled_clients / client_dir.name
                client_out.mkdir(exist_ok=True)
                
                for split in ['train', 'val', 'test']:
                    original = client_dir / f"{split}.parquet"
                    if original.exists():
                        df = pd.read_parquet(original)
                        sampled = df.sample(frac=min(fraction, 1.0), random_state=seed)
                        sampled.to_parquet(client_out / f"{split}.parquet")
                
                print(f"  Client {client_dir.name}: sampled")
    
    print(f"\nSampled data saved to: {sampled_dir}")
    return sampled_dir


def main():
    print("""
    ================================================================
    |  FL-TTA-IoT-IDS: Quick Validation on Lightning AI L40S      |
    |  1/4th Training for Fast Iteration                           |
    ================================================================
    """)
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "l40s_quick.yaml"
    config = load_config(str(config_path))
    
    # Setup L40S optimizations
    if torch.cuda.is_available():
        setup_l40s_optimizations()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"VRAM: {vram:.1f} GB")
        
        if 'L40S' not in gpu_name and 'A100' not in gpu_name:
            print("\nWARNING: Not running on L40S/A100. Performance may differ.")
    else:
        print("\nWARNING: No GPU detected. Training will be slow.")
    
    # Sample data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please run one of these first:")
        print("  python test_real_iot23.py")
        print("  python scripts/download_iot23.py --subset small")
        print("  python scripts/preprocess.py")
        return 1
    
    fraction = config.get('data_fraction', 0.25) if not args.debug else 0.1
    sampled_dir = sample_data(data_dir, fraction=fraction)
    
    # Setup WandB logger if enabled
    if not args.debug and not args.skip_wandb:
        try:
            import wandb
            wandb_config = config.get('wandb', {})
            
            wandb.init(
                project=wandb_config.get('project', 'fl-tta-iot-ids'),
                entity=os.environ.get('WANDB_ENTITY'),
                name=wandb_config.get('name', f"l40s_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config={
                    **config,
                    'data_dir': str(sampled_dir),
                    'mode': args.mode,
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
                }
            )
            print("\n✓ WandB logging enabled")
        except Exception as e:
            print(f"\nWARNING: WandB setup failed: {e}")
            print("Continuing without WandB logging...")
    
    # Create args for train functions
    class TrainArgs:
        def __init__(self):
            self.data_dir = str(sampled_dir)
            self.output_dir = "experiments_quick"
            self.max_epochs = config.get('max_epochs', 15)
            self.batch_size = config.get('batch_size', 4096)
            self.seed = 42
            self.debug = args.debug
            self.fl_rounds = config.get('federated', {}).get('num_rounds', 20)
            self.fl_epochs = config.get('federated', {}).get('local_epochs', 3)
            self.tta = 'drift_aware'
            self.tta_method = 'drift_aware'
            self.config = str(config_path)
            self.mode = args.mode
    
    train_args = TrainArgs()
    
    # Run training
    if args.mode == "centralized":
        model, results = train_centralized(train_args, config)
    elif args.mode in ["federated", "federated_tta"]:
        model, results = train_federated(
            train_args, 
            config, 
            with_tta=(args.mode == "federated_tta")
        )
    
    # Log to WandB
    if not args.debug and not args.skip_wandb:
        try:
            wandb.log({"final_results": results})
            wandb.finish()
        except:
            pass
    
    print("\n" + "="*60)
    print("QUICK VALIDATION COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
