"""
Main Training Script for FL-TTA IoT IDS
Optimized for Lightning AI L40S GPU (48GB VRAM)

Usage:
    # Test with tiny data
    python train.py --config configs/test_tiny.yaml
    
    # Full training on L40S
    python train.py --config configs/l40s_full.yaml
    
    # Centralized baseline
    python train.py --mode centralized --config configs/centralized.yaml
    
    # Federated + TTA
    python train.py --mode federated --tta tent --config configs/federated_tta.yaml
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import IoT23DataModule, FederatedClientDataModule, IoT23Dataset
from src.models import IoTIDSLightningModule, create_model
from src.fl import run_federated_simulation, get_parameters, set_parameters
from src.tta import TENT, EATA, DriftAwareTTA, create_tta


def parse_args():
    parser = argparse.ArgumentParser(description="FL-TTA IoT IDS Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, default="centralized",
                        choices=["centralized", "federated", "federated_tta"],
                        help="Training mode")
    parser.add_argument("--tta", type=str, default=None,
                        choices=["tent", "eata", "drift_aware"],
                        help="TTA method (only for federated_tta mode)")
    parser.add_argument("--data_dir", type=str, default="data/processed/iot23",
                        help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (fast dev run)")
    parser.add_argument("--fl_rounds", type=int, default=None,
                        help="Override FL rounds from config")
    parser.add_argument("--fl_epochs", type=int, default=None,
                        help="Override FL local epochs from config")
    parser.add_argument("--tta_method", type=str, default=None,
                        choices=["tent", "eata", "drift_aware"],
                        help="TTA method (alternative to --tta)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def setup_l40s_optimizations():
    """Configure optimizations for L40S GPU."""
    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True
    
    # Set memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    print("L40S optimizations enabled: TF32, cuDNN benchmark, Flash Attention")


def train_centralized(args, config: dict):
    """Centralized training baseline."""
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING")
    print("="*60 + "\n")
    
    # Data
    batch_size = args.batch_size or config.get('batch_size', 4096)
    data_module = IoT23DataModule(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=config.get('num_workers', 8),
        label_column=config.get('label_column', 'label_binary'),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    data_module.setup()
    
    # Model
    model_config = config.get('model', {})
    arch = model_config.pop('architecture', 'mlp')  # Remove to avoid duplicate
    model = IoTIDSLightningModule(
        input_dim=data_module.num_features,
        num_classes=data_module.num_classes,
        architecture=arch,
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        warmup_epochs=config.get('warmup_epochs', 5),
        max_epochs=args.max_epochs or config.get('max_epochs', 50),
        **model_config
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"centralized_{timestamp}"
    
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.output_dir}/{run_name}/checkpoints",
            filename="best-{epoch:02d}-{val_f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val/f1",
            patience=config.get('patience', 10),
            mode="max"
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar()
    ]
    
    # Logger
    logger = [
        TensorBoardLogger(args.output_dir, name=run_name),
        CSVLogger(args.output_dir, name=run_name)
    ]
    
    # Trainer (L40S optimized)
    trainer = L.Trainer(
        max_epochs=args.max_epochs or config.get('max_epochs', 50),
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",  # BF16 for L40S
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        log_every_n_steps=10,
        fast_dev_run=args.debug,
        enable_progress_bar=True,
        deterministic=False  # Faster without determinism
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test - try best checkpoint, fall back to last if none saved
    try:
        results = trainer.test(model, data_module, ckpt_path="best")
    except Exception as e:
        print(f"Note: Could not load best checkpoint ({e}), using current model")
        results = trainer.test(model, data_module)
    
    # Save results
    results_path = Path(args.output_dir) / run_name / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return model, results


def train_federated(args, config: dict, with_tta: bool = False):
    """Federated training with optional TTA."""
    print("\n" + "="*60)
    print(f"FEDERATED TRAINING {'+ TTA' if with_tta else ''}")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load feature columns
    data_dir = Path(args.data_dir)
    with open(data_dir / "feature_columns.txt", 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    # Load client data
    clients_dir = data_dir / "clients"
    client_dirs = list(clients_dir.iterdir())
    print(f"Found {len(client_dirs)} clients")
    
    # Create client data loaders
    batch_size = args.batch_size or config.get('batch_size', 2048)
    label_column = config.get('label_column', 'label_binary')
    
    # First pass: fit global scaler on combined train data
    print("Fitting global scaler...")
    all_train_dfs = []
    import pandas as pd
    for client_dir in client_dirs:
        train_path = client_dir / "train.parquet"
        if train_path.exists():
            df = pd.read_parquet(train_path)
            all_train_dfs.append(df[feature_columns])
    
    combined_features = pd.concat(all_train_dfs, ignore_index=True).values
    from sklearn.preprocessing import StandardScaler
    global_scaler = StandardScaler()
    global_scaler.fit(combined_features)
    del combined_features, all_train_dfs
    
    # Create client dataloaders
    client_data = {}
    for client_dir in client_dirs:
        client_name = client_dir.name
        
        train_dataset = IoT23Dataset(
            client_dir / "train.parquet",
            feature_columns,
            label_column,
            scaler=global_scaler
        )
        val_dataset = IoT23Dataset(
            client_dir / "val.parquet",
            feature_columns,
            label_column,
            scaler=global_scaler
        )
        
        # Adjust batch size for small datasets to avoid empty dataloaders
        client_batch_size = min(batch_size, len(train_dataset) // 2) if len(train_dataset) > 0 else batch_size
        client_batch_size = max(client_batch_size, 1)  # At least 1
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=client_batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=len(train_dataset) > client_batch_size
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=min(batch_size * 2, max(1, len(val_dataset))), shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        client_data[client_name] = (train_loader, val_loader)
        print(f"  {client_name}: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create global model
    model_config = config.get('model', {}).copy()  # Copy to avoid modifying config
    arch = model_config.pop('architecture', 'mlp')  # Remove to avoid duplicate
    num_classes = 2 if label_column == 'label_binary' else len(set())
    
    model = create_model(
        input_dim=len(feature_columns),
        num_classes=num_classes,
        architecture=arch,
        **model_config
    ).to(device)
    
    # Federated training
    fl_config = config.get('federated', {})
    fl_rounds = args.fl_rounds or fl_config.get('num_rounds', 50)
    fl_epochs = args.fl_epochs or fl_config.get('local_epochs', 5)
    
    model, history = run_federated_simulation(
        model=model,
        client_data=client_data,
        num_rounds=fl_rounds,
        local_epochs=fl_epochs,
        learning_rate=config.get('learning_rate', 1e-3),
        proximal_mu=fl_config.get('proximal_mu', 0.0),
        device=device,
        save_dir=Path(args.output_dir) / f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # TTA evaluation on test data
    tta_method_name = args.tta or args.tta_method
    if with_tta and tta_method_name:
        print("\n" + "="*60)
        print(f"TEST-TIME ADAPTATION ({tta_method_name.upper()})")
        print("="*60 + "\n")
        
        tta_config = config.get('tta', {})
        
        # Evaluate each client with TTA
        results_by_client = {}
        
        for client_name, client_dir in [(d.name, d) for d in client_dirs]:
            print(f"\nEvaluating {client_name} with TTA...")
            
            test_dataset = IoT23Dataset(
                client_dir / "test.parquet",
                feature_columns,
                label_column,
                scaler=global_scaler
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            
            # Create TTA wrapper
            import copy
            client_model = copy.deepcopy(model)
            
            tta_config = config.get('tta', {}).copy()  # Copy to avoid modifying
            tta_lr = tta_config.pop('lr', 1e-4)
            tta_steps = tta_config.pop('steps', 1)
            
            tta_method = create_tta(
                client_model,
                method=tta_method_name,
                lr=tta_lr,
                steps=tta_steps
            )
            
            # Set baseline for drift-aware TTA
            if tta_method_name == 'drift_aware':
                val_loader = client_data[client_name][1]
                tta_method.set_baseline(val_loader)
            
            # Evaluate with TTA
            all_preds = []
            all_labels = []
            
            for x, y in test_loader:
                x = x.to(device)
                logits = tta_method(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
            
            # Compute metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            results_by_client[client_name] = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
                'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
                'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
            }
            
            print(f"  Accuracy: {results_by_client[client_name]['accuracy']:.4f}")
            print(f"  F1 Macro: {results_by_client[client_name]['f1_macro']:.4f}")
            
            if hasattr(tta_method, 'get_stats'):
                stats = tta_method.get_stats()
                print(f"  TTA Stats: {stats}")
        
        # Aggregate results
        avg_acc = sum(r['accuracy'] for r in results_by_client.values()) / len(results_by_client)
        avg_f1 = sum(r['f1_macro'] for r in results_by_client.values()) / len(results_by_client)
        
        print("\n" + "="*60)
        print("FINAL RESULTS (with TTA)")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average F1 Macro: {avg_f1:.4f}")
        print("="*60)
        
        history['tta_results'] = results_by_client
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"federated_{'tta_' if with_tta else ''}{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "history.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for k, v in history.items():
            if isinstance(v, list) and len(v) > 0:
                if hasattr(v[0], 'tolist'):
                    serializable_history[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
                else:
                    serializable_history[k] = v
            else:
                serializable_history[k] = v
        json.dump(serializable_history, f, indent=2, default=str)
    
    torch.save(model.state_dict(), results_dir / "final_model.pt")
    
    print(f"\nResults saved to: {results_dir}")
    
    return model, history


def main():
    args = parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Setup L40S optimizations
    if torch.cuda.is_available():
        setup_l40s_optimizations()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    if args.mode == "centralized":
        model, results = train_centralized(args, config)
    elif args.mode == "federated":
        model, results = train_federated(args, config, with_tta=False)
    elif args.mode == "federated_tta":
        model, results = train_federated(args, config, with_tta=True)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
