#!/usr/bin/env python3
"""
=============================================================================
LIGHTNING AI L40S QUICK TRAINING SCRIPT (1/4th Training)
=============================================================================
Optimized for ~15-20 min training on L40S GPU ($1.49/hr = ~$0.50 cost)

Features:
- WandB integration for experiment tracking
- Full L40S GPU utilization (48GB VRAM)
- Downloads real IoT-23 data automatically
- Federated Learning + Drift-Aware TTA
- Exports results and model checkpoints

Usage on Lightning AI:
    # Set your WandB API key first
    export WANDB_API_KEY=your_key_here
    
    # Run training
    python train_l40s_quarter.py --wandb_key YOUR_KEY
    
    # Or run without WandB
    python train_l40s_quarter.py --no_wandb
=============================================================================
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))


def setup_environment():
    """Setup L40S optimizations and imports."""
    import torch
    
    # L40S GPU Optimizations
    if torch.cuda.is_available():
        # Enable TF32 for massive speedup on Ampere/Ada GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable Flash Attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
        print("L40S optimizations enabled: TF32, cuDNN benchmark, Flash Attention")
    else:
        print("WARNING: No GPU detected! Training will be slow.")
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_wandb(api_key: str, project: str = "FL-TTA-IoT-IDS", run_name: str = None):
    """Initialize WandB for experiment tracking."""
    try:
        import wandb
        
        if api_key:
            wandb.login(key=api_key)
        
        run_name = run_name or f"l40s_quarter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = wandb.init(
            project=project,
            name=run_name,
            config={
                "training_type": "quarter",
                "gpu": "L40S",
                "batch_size": 16384,
                "fl_rounds": 20,
                "local_epochs": 3,
                "tta_method": "drift_aware",
            },
            tags=["l40s", "quarter", "fl", "tta", "iot23"]
        )
        print(f"WandB run initialized: {run.url}")
        return run
    except ImportError:
        print("WandB not installed. Run: pip install wandb")
        return None
    except Exception as e:
        print(f"WandB setup failed: {e}")
        return None


def download_iot23_data(output_dir: Path, num_scenarios: int = 3):
    """Download real IoT-23 data for training."""
    import urllib.request
    import ssl
    
    print("\n" + "="*60)
    print("DOWNLOADING REAL IOT-23 DATA")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Best scenarios for quick but meaningful training
    scenarios = {
        "CTU-IoT-Malware-Capture-44-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-44-1/bro/conn.log.labeled",
        "CTU-IoT-Malware-Capture-8-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled",
        "CTU-IoT-Malware-Capture-34-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled",
    }
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    downloaded = []
    for i, (name, url) in enumerate(list(scenarios.items())[:num_scenarios]):
        output_file = output_dir / f"{name}_conn.log.labeled"
        
        if output_file.exists():
            print(f"[{i+1}/{num_scenarios}] {name}: Already exists")
            downloaded.append(output_file)
            continue
        
        print(f"[{i+1}/{num_scenarios}] Downloading {name}...")
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as response:
                data = response.read()
                with open(output_file, 'wb') as f:
                    f.write(data)
            size_kb = output_file.stat().st_size / 1024
            print(f"    Downloaded: {size_kb:.1f} KB")
            downloaded.append(output_file)
        except Exception as e:
            print(f"    Failed: {e}")
    
    return downloaded


def preprocess_data(raw_files: list, output_dir: Path):
    """Preprocess IoT-23 data for training."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    for filepath in raw_files:
        print(f"Parsing: {filepath.name}")
        
        # Parse Zeek log
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                data_start = i
                break
        
        # Default columns
        columns = [
            'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
            'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
            'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
            'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'tunnel_parents', 'label', 'detailed-label'
        ]
        
        rows = []
        for line in lines[data_start:]:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= len(columns):
                rows.append(parts[:len(columns)])
            elif len(parts) > 10:
                parts.extend([''] * (len(columns) - len(parts)))
                rows.append(parts)
        
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            df['scenario'] = filepath.stem.replace('_conn.log.labeled', '')
            all_dfs.append(df)
            print(f"  Loaded {len(df)} flows")
    
    if not all_dfs:
        raise ValueError("No data loaded!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal flows: {len(combined)}")
    
    # Create labels
    combined['label_binary'] = (~combined['label'].str.contains('Benign', case=False, na=False)).astype(int)
    
    # Numeric features
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
                    'id.orig_p', 'id.resp_p']
    
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col].replace('-', '0'), errors='coerce').fillna(0)
    
    # Engineered features
    combined['bytes_ratio'] = (combined['orig_bytes'] + 1) / (combined['resp_bytes'] + 1)
    combined['pkts_ratio'] = (combined['orig_pkts'] + 1) / (combined['resp_pkts'] + 1)
    combined['bytes_per_pkt_orig'] = combined['orig_bytes'] / (combined['orig_pkts'] + 1)
    combined['bytes_per_pkt_resp'] = combined['resp_bytes'] / (combined['resp_pkts'] + 1)
    combined['duration_log'] = np.log1p(combined['duration'])
    combined['total_bytes'] = combined['orig_bytes'] + combined['resp_bytes']
    combined['total_pkts'] = combined['orig_pkts'] + combined['resp_pkts']
    combined['is_well_known_port'] = (combined['id.resp_p'] < 1024).astype(int)
    combined['is_high_port'] = (combined['id.resp_p'] > 49152).astype(int)
    combined['history_len'] = combined['history'].fillna('').str.len()
    
    # Categorical encoding
    for col in ['proto', 'service', 'conn_state']:
        if col in combined.columns:
            mapping = {v: i for i, v in enumerate(combined[col].fillna('unknown').unique())}
            combined[f'{col}_encoded'] = combined[col].fillna('unknown').map(mapping)
    
    # Feature columns
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'id.orig_p', 'id.resp_p',
        'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_orig', 'bytes_per_pkt_resp',
        'duration_log', 'total_bytes', 'total_pkts',
        'is_well_known_port', 'is_high_port', 'history_len',
        'proto_encoded', 'service_encoded', 'conn_state_encoded'
    ]
    feature_columns = [c for c in feature_columns if c in combined.columns]
    
    # Save feature columns
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Shuffle and split
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(combined)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    cols_to_save = feature_columns + ['label_binary', 'scenario']
    
    combined.iloc[:train_end][cols_to_save].to_parquet(output_dir / "train.parquet")
    combined.iloc[train_end:val_end][cols_to_save].to_parquet(output_dir / "val.parquet")
    combined.iloc[val_end:][cols_to_save].to_parquet(output_dir / "test.parquet")
    
    print(f"Saved: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    
    # Create client splits
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(exist_ok=True)
    
    for scenario in combined['scenario'].unique():
        scenario_df = combined[combined['scenario'] == scenario]
        sn = len(scenario_df)
        st_end = int(sn * 0.7)
        sv_end = int(sn * 0.85)
        
        client_dir = clients_dir / scenario.replace('-', '_')
        client_dir.mkdir(exist_ok=True)
        
        scenario_df.iloc[:st_end][cols_to_save].to_parquet(client_dir / "train.parquet")
        scenario_df.iloc[st_end:sv_end][cols_to_save].to_parquet(client_dir / "val.parquet")
        scenario_df.iloc[sv_end:][cols_to_save].to_parquet(client_dir / "test.parquet")
        
        print(f"  {scenario}: {st_end}/{sv_end-st_end}/{sn-sv_end}")
    
    return output_dir, feature_columns


def train_federated_l40s(data_dir: Path, config: dict, wandb_run=None):
    """Run optimized federated training for L40S."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    
    print("\n" + "="*60)
    print("FEDERATED TRAINING (L40S OPTIMIZED)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load feature columns
    with open(data_dir / "feature_columns.txt", 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    # Load and preprocess client data
    clients_dir = data_dir / "clients"
    client_dirs = list(clients_dir.iterdir())
    print(f"Found {len(client_dirs)} clients")
    
    # Fit global scaler
    all_train = []
    for client_dir in client_dirs:
        df = pd.read_parquet(client_dir / "train.parquet")
        all_train.append(df[feature_columns].values)
    
    combined_features = np.vstack(all_train)
    scaler = StandardScaler()
    scaler.fit(combined_features)
    del combined_features, all_train
    
    # Prepare client dataloaders
    batch_size = config.get('batch_size', 16384)
    client_data = {}
    
    for client_dir in client_dirs:
        name = client_dir.name
        
        # Train data
        train_df = pd.read_parquet(client_dir / "train.parquet")
        train_X = scaler.transform(train_df[feature_columns].values.astype(np.float32))
        train_X = np.nan_to_num(train_X, 0)
        train_y = train_df['label_binary'].values.astype(np.int64)
        
        # Val data
        val_df = pd.read_parquet(client_dir / "val.parquet")
        val_X = scaler.transform(val_df[feature_columns].values.astype(np.float32))
        val_X = np.nan_to_num(val_X, 0)
        val_y = val_df['label_binary'].values.astype(np.int64)
        
        # Adaptive batch size for small clients
        client_batch = min(batch_size, max(1, len(train_X) // 2))
        
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)),
            batch_size=client_batch, shuffle=True, num_workers=4, pin_memory=True,
            drop_last=len(train_X) > client_batch
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y)),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        client_data[name] = (train_loader, val_loader, len(train_X))
        print(f"  {name}: {len(train_X)} train, {len(val_X)} val")
    
    # Create model
    input_dim = len(feature_columns)
    hidden_dims = config.get('model', {}).get('hidden_dims', [768, 384, 192])
    dropout = config.get('model', {}).get('dropout', 0.15)
    
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = h_dim
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(prev_dim, 2)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            return self.classifier(self.features(x))
    
    global_model = MLP().to(device)
    
    # FL settings
    num_rounds = config.get('federated', {}).get('num_rounds', 20)
    local_epochs = config.get('federated', {}).get('local_epochs', 3)
    proximal_mu = config.get('federated', {}).get('proximal_mu', 0.01)
    lr = config.get('learning_rate', 0.003)
    
    # Training loop
    history = {'round': [], 'loss': [], 'accuracy': [], 'f1': []}
    best_f1 = 0
    
    for round_num in range(num_rounds):
        round_start = time.time()
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Store global params for FedProx
        global_params = [p.clone().detach() for p in global_model.parameters()]
        
        # Client training
        client_updates = []
        
        for client_name, (train_loader, val_loader, n_samples) in client_data.items():
            # Copy global model
            local_model = MLP().to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
            
            # Local training
            for epoch in range(local_epochs):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    loss = F.cross_entropy(local_model(x), y)
                    
                    # FedProx regularization
                    if proximal_mu > 0:
                        prox_loss = 0
                        for p, gp in zip(local_model.parameters(), global_params):
                            prox_loss += ((p - gp) ** 2).sum()
                        loss += (proximal_mu / 2) * prox_loss
                    
                    loss.backward()
                    optimizer.step()
            
            # Collect update
            client_updates.append((
                [p.cpu().detach().numpy() for p in local_model.parameters()],
                n_samples
            ))
        
        # FedAvg aggregation
        total_samples = sum(n for _, n in client_updates)
        new_params = None
        
        for params, n_samples in client_updates:
            weight = n_samples / total_samples
            if new_params is None:
                new_params = [weight * p for p in params]
            else:
                for i, p in enumerate(params):
                    new_params[i] += weight * p
        
        # Update global model
        with torch.no_grad():
            for p, new_p in zip(global_model.parameters(), new_params):
                p.copy_(torch.from_numpy(new_p).to(device))
        
        # Evaluate
        global_model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            for client_name, (_, val_loader, _) in client_data.items():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = global_model(x)
                    total_loss += F.cross_entropy(logits, y).item()
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        round_time = time.time() - round_start
        print(f"  Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Time: {round_time:.1f}s")
        
        # Log to WandB
        if wandb_run:
            import wandb
            wandb.log({
                'round': round_num + 1,
                'val_loss': total_loss,
                'val_accuracy': acc,
                'val_f1': f1,
                'round_time': round_time
            })
        
        history['round'].append(round_num + 1)
        history['loss'].append(total_loss)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = global_model.state_dict()
    
    print(f"\n Best F1: {best_f1:.4f}")
    global_model.load_state_dict(best_model_state)
    
    return global_model, scaler, history, feature_columns


def evaluate_with_tta(model, scaler, data_dir: Path, feature_columns: list, config: dict, wandb_run=None):
    """Evaluate with Drift-Aware TTA."""
    import torch
    import torch.nn.functional as F
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from collections import deque
    
    print("\n" + "="*60)
    print("DRIFT-AWARE TTA EVALUATION")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Load test data
    test_df = pd.read_parquet(data_dir / "test.parquet")
    test_X = scaler.transform(test_df[feature_columns].values.astype(np.float32))
    test_X = np.nan_to_num(test_X, 0)
    test_y = test_df['label_binary'].values
    
    print(f"Test samples: {len(test_X)}")
    
    # Compute baseline entropy from validation
    val_df = pd.read_parquet(data_dir / "val.parquet")
    val_X = scaler.transform(val_df[feature_columns].values.astype(np.float32))
    val_X = np.nan_to_num(val_X, 0)
    
    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(val_X).to(device)
        val_logits = model(val_tensor)
        val_probs = F.softmax(val_logits, dim=1)
        val_entropy = -(val_probs * torch.log(val_probs + 1e-8)).sum(dim=1)
        baseline_entropy = val_entropy.mean().item()
    
    print(f"Baseline entropy: {baseline_entropy:.4f}")
    
    # TTA settings
    tta_lr = config.get('tta', {}).get('lr', 0.0001)
    drift_threshold = config.get('tta', {}).get('entropy_drift_threshold', 0.3)
    entropy_threshold = config.get('tta', {}).get('entropy_threshold', 0.28)
    
    # Collect BN parameters
    bn_params = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            for param in module.parameters():
                if param.requires_grad:
                    bn_params.append(param)
    
    optimizer = torch.optim.Adam(bn_params, lr=tta_lr)
    
    # Process in batches
    batch_size = 512
    entropy_history = deque(maxlen=50)
    all_preds = []
    adaptation_count = 0
    
    for i in range(0, len(test_X), batch_size):
        batch_X = torch.from_numpy(test_X[i:i+batch_size]).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(batch_X)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            batch_entropy = entropy.mean().item()
        
        entropy_history.append(batch_entropy)
        
        # Check for drift
        if len(entropy_history) >= 10:
            window_entropy = np.mean(list(entropy_history))
            relative_increase = (window_entropy - baseline_entropy) / (baseline_entropy + 1e-6)
            
            if relative_increase > drift_threshold:
                # Drift detected - apply TTA
                model.train()
                
                # Only adapt on reliable samples
                reliable_mask = entropy < entropy_threshold
                if reliable_mask.sum() > 0:
                    optimizer.zero_grad()
                    logits = model(batch_X)
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                    loss = entropy[reliable_mask].mean()
                    loss.backward()
                    optimizer.step()
                    adaptation_count += 1
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            logits = model(batch_X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    # Compute metrics
    acc = accuracy_score(test_y, all_preds)
    f1 = f1_score(test_y, all_preds, average='macro', zero_division=0)
    
    print(f"\nTTA Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Adaptations: {adaptation_count}")
    print(f"\nClassification Report:")
    print(classification_report(test_y, all_preds, target_names=['Benign', 'Malicious']))
    
    if wandb_run:
        import wandb
        wandb.log({
            'test_accuracy': acc,
            'test_f1': f1,
            'tta_adaptations': adaptation_count
        })
        
        # Log confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_y, all_preds)
        wandb.log({
            'confusion_matrix': wandb.plot.confusion_matrix(
                y_true=test_y, preds=all_preds,
                class_names=['Benign', 'Malicious']
            )
        })
    
    return {'accuracy': acc, 'f1': f1, 'adaptations': adaptation_count}


def main():
    parser = argparse.ArgumentParser(description="L40S Quarter Training")
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--data_dir', type=str, default='data/processed/iot23_l40s', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/l40s_quarter', help='Output directory')
    parser.add_argument('--num_scenarios', type=int, default=3, help='Number of IoT-23 scenarios')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     FL-TTA-IoT-IDS: L40S QUARTER TRAINING                    ║
    ║     Optimized for ~15-20 min on Lightning AI ($0.50 cost)    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    device = setup_environment()
    
    # WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_key = args.wandb_key or os.environ.get('WANDB_API_KEY')
        if wandb_key:
            wandb_run = setup_wandb(wandb_key)
        else:
            print("No WandB API key provided. Run with --wandb_key YOUR_KEY or set WANDB_API_KEY")
            print("Continuing without WandB...")
    
    # Download data
    raw_dir = Path(args.data_dir).parent / "raw_iot23"
    downloaded_files = download_iot23_data(raw_dir, args.num_scenarios)
    
    if not downloaded_files:
        print("ERROR: No data downloaded!")
        return 1
    
    # Preprocess
    data_dir, feature_columns = preprocess_data(downloaded_files, Path(args.data_dir))
    
    # Load config
    config = {
        'batch_size': 16384,
        'learning_rate': 0.003,
        'model': {'hidden_dims': [768, 384, 192], 'dropout': 0.15},
        'federated': {'num_rounds': 20, 'local_epochs': 3, 'proximal_mu': 0.01},
        'tta': {'lr': 0.0001, 'entropy_threshold': 0.28, 'entropy_drift_threshold': 0.3}
    }
    
    # Train
    model, scaler, history, feature_columns = train_federated_l40s(
        data_dir, config, wandb_run
    )
    
    # Evaluate with TTA
    results = evaluate_with_tta(model, scaler, data_dir, feature_columns, config, wandb_run)
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import torch
    torch.save(model.state_dict(), output_dir / "model_best.pt")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'history': history,
            'test_results': results,
            'training_time': time.time() - start_time
        }, f, indent=2)
    
    total_time = time.time() - start_time
    estimated_cost = (total_time / 3600) * 1.49
    
    print(f"\n" + "="*60)
    print(f"TRAINING COMPLETE!")
    print(f"="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Best F1: {max(history['f1']):.4f}")
    print(f"Test F1: {results['f1']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    if wandb_run:
        import wandb
        wandb.finish()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
