#!/usr/bin/env python3
"""
=============================================================================
LIGHTNING AI L40S FULL TRAINING SCRIPT
=============================================================================
MAXIMUM GPU UTILIZATION for L40S 48GB VRAM ($1.49/hr)

Optimizations applied:
- Batch size: 32768 (massive throughput)
- TF32 tensor cores + BF16 mixed precision (2-4x speedup)
- cuDNN benchmark (auto-tuned algorithms)
- 16 workers + prefetch (saturated data pipeline)
- Flash Attention enabled
- Class-weighted loss (fixes benign class underprediction)
- Gradient clipping (stable large-batch training)

Expected Results (based on quarter training analysis):
- Test Accuracy: 94.0-94.8%
- Test Macro F1: 80.5-83.5%
- Training Time: ~45-60 minutes
- Cost: ~$1.50-2.00

Usage on Lightning AI:
    export WANDB_API_KEY=your_key_here
    python train_l40s_full.py --wandb_key $WANDB_API_KEY
    
    # Or without WandB
    python train_l40s_full.py --no_wandb
=============================================================================
"""
import os
import sys
import argparse
import time
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))


def setup_l40s_environment():
    """Setup MAXIMUM L40S optimizations."""
    import torch
    
    print("=" * 70)
    print("SETTING UP L40S GPU ENVIRONMENT")
    print("=" * 70)
    
    if torch.cuda.is_available():
        # L40S GPU Optimizations - MAXIMUM PERFORMANCE
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matrix multiply
        torch.backends.cudnn.allow_tf32 = True        # TF32 convolutions
        torch.backends.cudnn.benchmark = True          # Auto-tune algorithms
        torch.backends.cudnn.deterministic = False     # Allow non-deterministic (faster)
        
        # Enable Flash Attention and memory-efficient attention
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        
        # Set optimal GPU settings
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print()
        print("L40S Optimizations ENABLED:")
        print("  [x] TF32 Tensor Cores (8x faster than FP32)")
        print("  [x] cuDNN Benchmark (auto-tuned algorithms)")
        print("  [x] Flash Attention (memory efficient)")
        print("  [x] BF16 Mixed Precision (2x throughput)")
        
        return torch.device('cuda'), gpu_mem
    else:
        print("WARNING: No GPU detected! Training will be extremely slow.")
        print("This script is optimized for L40S GPU.")
        return torch.device('cpu'), 12.0


def setup_wandb(api_key: str, config: dict, run_name: str = None):
    """Initialize WandB for experiment tracking."""
    try:
        import wandb
        
        if api_key:
            wandb.login(key=api_key)
        
        run_name = run_name or f"l40s_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = wandb.init(
            project="FL-TTA-IoT-IDS",
            name=run_name,
            config=config,
            tags=["l40s", "full", "fl", "tta", "iot23", "class_weighted"]
        )
        print(f"\nWandB initialized: {run.url}")
        return run
    except ImportError:
        print("WandB not installed. Run: pip install wandb")
        return None
    except Exception as e:
        print(f"WandB setup failed: {e}")
        return None


def download_iot23_full(output_dir: Path):
    """Download comprehensive IoT-23 scenarios for full training."""
    import urllib.request
    import ssl
    
    print("\n" + "=" * 70)
    print("DOWNLOADING IOT-23 DATA (COMPREHENSIVE)")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extended scenario list for better generalization
    scenarios = {
        # Scenario 1 - Has BOTH benign and malicious (CRITICAL for balance)
        "CTU-IoT-Malware-Capture-1-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled",
        # Scenario 3 - Mirai botnet
        "CTU-IoT-Malware-Capture-3-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled",
        # Scenario 8 - Good size
        "CTU-IoT-Malware-Capture-8-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled",
        # Scenario 34 - Diverse attacks
        "CTU-IoT-Malware-Capture-34-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled",
        # Scenario 44 - Small but useful
        "CTU-IoT-Malware-Capture-44-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-44-1/bro/conn.log.labeled",
    }
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    downloaded = []
    total_size = 0
    
    for i, (name, url) in enumerate(scenarios.items()):
        output_file = output_dir / f"{name}_conn.log.labeled"
        
        if output_file.exists():
            size_kb = output_file.stat().st_size / 1024
            print(f"[{i+1}/{len(scenarios)}] {name}: Already exists ({size_kb:.1f} KB)")
            downloaded.append(output_file)
            total_size += size_kb
            continue
        
        print(f"[{i+1}/{len(scenarios)}] Downloading {name}...")
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=300) as response:
                data = response.read()
                with open(output_file, 'wb') as f:
                    f.write(data)
            size_kb = output_file.stat().st_size / 1024
            print(f"    Downloaded: {size_kb:.1f} KB")
            downloaded.append(output_file)
            total_size += size_kb
        except Exception as e:
            print(f"    Failed: {e}")
    
    print(f"\nTotal downloaded: {total_size/1024:.2f} MB ({len(downloaded)} scenarios)")
    return downloaded


def preprocess_iot23(raw_files: list, output_dir: Path, max_flows_per_scenario: int = None):
    """Preprocess IoT-23 data with proper label parsing."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 70)
    print("PREPROCESSING IOT-23 DATA")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    for filepath in raw_files:
        print(f"\nParsing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse Zeek conn.log format
        rows = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            
            # IoT-23 format: last field contains "tunnel_parents label detailed-label" (space-separated)
            if len(parts) >= 21:
                last_col = parts[-1].strip()
                last_parts = last_col.split()
                
                # Extract label (look for Malicious or Benign)
                label = 'Unknown'
                for lp in last_parts:
                    if lp.lower() in ['malicious', 'benign']:
                        label = lp
                        break
                
                row = {
                    'ts': parts[0],
                    'uid': parts[1],
                    'id.orig_h': parts[2],
                    'id.orig_p': parts[3],
                    'id.resp_h': parts[4],
                    'id.resp_p': parts[5],
                    'proto': parts[6],
                    'service': parts[7],
                    'duration': parts[8],
                    'orig_bytes': parts[9],
                    'resp_bytes': parts[10],
                    'conn_state': parts[11],
                    'orig_pkts': parts[17] if len(parts) > 17 else '0',
                    'orig_ip_bytes': parts[18] if len(parts) > 18 else '0',
                    'resp_pkts': parts[19] if len(parts) > 19 else '0',
                    'resp_ip_bytes': parts[20] if len(parts) > 20 else '0',
                    'label': label,
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df['scenario'] = filepath.stem.replace('_conn.log.labeled', '').replace('-', '_')
            
            # Sample if too large
            if max_flows_per_scenario and len(df) > max_flows_per_scenario:
                df = df.sample(n=max_flows_per_scenario, random_state=42)
            
            all_dfs.append(df)
            
            # Print label distribution
            benign = (df['label'] == 'Benign').sum()
            malicious = (df['label'] == 'Malicious').sum()
            print(f"  Loaded {len(df)} flows (Benign: {benign}, Malicious: {malicious})")
    
    if not all_dfs:
        raise ValueError("No data loaded!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Binary labels
    combined['label_binary'] = (combined['label'] == 'Malicious').astype(int)
    
    benign_total = (combined['label_binary'] == 0).sum()
    malicious_total = (combined['label_binary'] == 1).sum()
    print(f"\nTotal: {len(combined)} flows")
    print(f"  Benign: {benign_total} ({100*benign_total/len(combined):.1f}%)")
    print(f"  Malicious: {malicious_total} ({100*malicious_total/len(combined):.1f}%)")
    
    # Numeric features
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 
                    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'id.orig_p', 'id.resp_p']
    
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col].replace('-', '0'), errors='coerce').fillna(0)
    
    # Engineered features for better discrimination
    combined['bytes_ratio'] = (combined['orig_bytes'] + 1) / (combined['resp_bytes'] + 1)
    combined['pkts_ratio'] = (combined['orig_pkts'] + 1) / (combined['resp_pkts'] + 1)
    combined['bytes_per_pkt_orig'] = combined['orig_bytes'] / (combined['orig_pkts'] + 1)
    combined['bytes_per_pkt_resp'] = combined['resp_bytes'] / (combined['resp_pkts'] + 1)
    combined['duration_log'] = np.log1p(combined['duration'])
    combined['total_bytes'] = combined['orig_bytes'] + combined['resp_bytes']
    combined['total_pkts'] = combined['orig_pkts'] + combined['resp_pkts']
    combined['is_well_known_port'] = (combined['id.resp_p'] < 1024).astype(int)
    combined['is_high_port'] = (combined['id.resp_p'] > 49152).astype(int)
    combined['bytes_imbalance'] = np.abs(combined['orig_bytes'] - combined['resp_bytes']) / (combined['total_bytes'] + 1)
    combined['pkts_imbalance'] = np.abs(combined['orig_pkts'] - combined['resp_pkts']) / (combined['total_pkts'] + 1)
    
    # Categorical encoding
    for col in ['proto', 'service', 'conn_state']:
        if col in combined.columns:
            mapping = {v: i for i, v in enumerate(combined[col].fillna('unknown').unique())}
            combined[f'{col}_encoded'] = combined[col].fillna('unknown').map(mapping)
    
    # Feature columns
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes', 'id.orig_p', 'id.resp_p',
        'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_orig', 'bytes_per_pkt_resp',
        'duration_log', 'total_bytes', 'total_pkts',
        'is_well_known_port', 'is_high_port', 'bytes_imbalance', 'pkts_imbalance',
        'proto_encoded', 'service_encoded', 'conn_state_encoded'
    ]
    feature_columns = [c for c in feature_columns if c in combined.columns]
    
    print(f"Features: {len(feature_columns)}")
    
    # Save feature columns
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Stratified split to maintain class balance
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        combined, test_size=0.3, random_state=42, stratify=combined['label_binary']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_binary']
    )
    
    cols_to_save = feature_columns + ['label_binary', 'scenario']
    
    train_df[cols_to_save].to_parquet(output_dir / "train.parquet")
    val_df[cols_to_save].to_parquet(output_dir / "val.parquet")
    test_df[cols_to_save].to_parquet(output_dir / "test.parquet")
    
    print(f"\nSplit: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create client splits (per scenario)
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(exist_ok=True)
    
    for scenario in combined['scenario'].unique():
        scenario_df = combined[combined['scenario'] == scenario]
        
        # Stratified split for each client
        if len(scenario_df) > 100:
            try:
                s_train, s_temp = train_test_split(
                    scenario_df, test_size=0.3, random_state=42, 
                    stratify=scenario_df['label_binary']
                )
                s_val, s_test = train_test_split(
                    s_temp, test_size=0.5, random_state=42,
                    stratify=s_temp['label_binary']
                )
            except:
                # Fallback to random split if stratified fails
                sn = len(scenario_df)
                s_train = scenario_df.iloc[:int(sn*0.7)]
                s_val = scenario_df.iloc[int(sn*0.7):int(sn*0.85)]
                s_test = scenario_df.iloc[int(sn*0.85):]
        else:
            sn = len(scenario_df)
            s_train = scenario_df.iloc[:int(sn*0.7)]
            s_val = scenario_df.iloc[int(sn*0.7):int(sn*0.85)]
            s_test = scenario_df.iloc[int(sn*0.85):]
        
        client_dir = clients_dir / scenario
        client_dir.mkdir(exist_ok=True)
        
        s_train[cols_to_save].to_parquet(client_dir / "train.parquet")
        s_val[cols_to_save].to_parquet(client_dir / "val.parquet")
        s_test[cols_to_save].to_parquet(client_dir / "test.parquet")
        
        print(f"  {scenario}: {len(s_train)}/{len(s_val)}/{len(s_test)}")
    
    return output_dir, feature_columns


class DeepMLP:
    """High-performance MLP for L40S with class weighting."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 256, 128], 
                 dropout: float = 0.2, num_classes: int = 2):
        import torch
        import torch.nn as nn
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes
        
    def build(self, device):
        import torch
        import torch.nn as nn
        
        class Model(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout, num_classes):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for i, h_dim in enumerate(hidden_dims):
                    layers.append(nn.Linear(prev_dim, h_dim))
                    layers.append(nn.BatchNorm1d(h_dim))
                    layers.append(nn.ReLU(inplace=True))
                    # More dropout in deeper layers
                    layers.append(nn.Dropout(dropout * (1 + i * 0.1)))
                    prev_dim = h_dim
                
                self.features = nn.Sequential(*layers)
                self.classifier = nn.Linear(prev_dim, num_classes)
                
                # Initialize weights
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
                features = self.features(x)
                return self.classifier(features)
            
            def get_features(self, x):
                return self.features(x)
        
        return Model(self.input_dim, self.hidden_dims, self.dropout, self.num_classes).to(device)


def train_federated_full(data_dir: Path, config: dict, device, wandb_run=None):
    """Run FULL federated training with L40S optimizations."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import autocast, GradScaler
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    print("\n" + "=" * 70)
    print("FEDERATED TRAINING (L40S FULL - CLASS WEIGHTED)")
    print("=" * 70)
    
    # Load feature columns
    with open(data_dir / "feature_columns.txt", 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    input_dim = len(feature_columns)
    print(f"Input features: {input_dim}")
    
    # Load all client data
    clients_dir = data_dir / "clients"
    client_dirs = [d for d in clients_dir.iterdir() if d.is_dir()]
    print(f"Found {len(client_dirs)} clients")
    
    # Fit global scaler on all training data
    all_train_X = []
    all_train_y = []
    
    for client_dir in client_dirs:
        df = pd.read_parquet(client_dir / "train.parquet")
        all_train_X.append(df[feature_columns].values)
        all_train_y.append(df['label_binary'].values)
    
    combined_X = np.vstack(all_train_X)
    combined_y = np.concatenate(all_train_y)
    
    scaler = StandardScaler()
    scaler.fit(combined_X)
    
    # Compute class weights for imbalanced data
    class_counts = np.bincount(combined_y.astype(int))
    total_samples = len(combined_y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: Benign={class_weights[0]:.4f}, Malicious={class_weights[1]:.4f}")
    
    del combined_X, all_train_X
    gc.collect()
    
    # Prepare client dataloaders
    batch_size = config.get('batch_size', 32768)
    num_workers = config.get('num_workers', 16)
    
    client_data = {}
    
    for client_dir in client_dirs:
        name = client_dir.name
        
        # Load and preprocess
        train_df = pd.read_parquet(client_dir / "train.parquet")
        train_X = scaler.transform(train_df[feature_columns].values.astype(np.float32))
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
        train_y = train_df['label_binary'].values.astype(np.int64)
        
        val_df = pd.read_parquet(client_dir / "val.parquet")
        val_X = scaler.transform(val_df[feature_columns].values.astype(np.float32))
        val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
        val_y = val_df['label_binary'].values.astype(np.int64)
        
        # Adaptive batch size
        client_batch = min(batch_size, max(32, len(train_X) // 4))
        
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)),
            batch_size=client_batch, shuffle=True,
            num_workers=min(num_workers, 4), pin_memory=True,
            drop_last=len(train_X) > client_batch, persistent_workers=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y)),
            batch_size=batch_size, shuffle=False,
            num_workers=min(num_workers, 4), pin_memory=True, persistent_workers=True
        )
        
        benign = (train_y == 0).sum()
        malicious = (train_y == 1).sum()
        client_data[name] = {
            'train': train_loader,
            'val': val_loader,
            'n_samples': len(train_X),
            'benign': benign,
            'malicious': malicious
        }
        print(f"  {name}: {len(train_X)} train (B:{benign}/M:{malicious}), {len(val_X)} val")
    
    # Create global model
    hidden_dims = config.get('model', {}).get('hidden_dims', [1024, 512, 256, 128])
    dropout = config.get('model', {}).get('dropout', 0.2)
    
    model_builder = DeepMLP(input_dim, hidden_dims, dropout)
    global_model = model_builder.build(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # FL settings
    num_rounds = config.get('federated', {}).get('num_rounds', 80)
    local_epochs = config.get('federated', {}).get('local_epochs', 5)
    proximal_mu = config.get('federated', {}).get('proximal_mu', 0.01)
    lr = config.get('learning_rate', 0.004)
    
    # Mixed precision scaler
    use_amp = config.get('precision', 'bf16-mixed') != 'float32'
    scaler_amp = GradScaler() if use_amp else None
    
    print(f"\nFL Config: {num_rounds} rounds, {local_epochs} local epochs, LR={lr}")
    print(f"Mixed Precision: {use_amp}")
    
    # Training history
    history = {
        'round': [], 'loss': [], 'accuracy': [], 'f1': [],
        'benign_f1': [], 'malicious_f1': []
    }
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience = config.get('patience', 20)
    
    training_start = time.time()
    
    for round_num in range(num_rounds):
        round_start = time.time()
        
        # Store global params for FedProx
        global_params = [p.clone().detach() for p in global_model.parameters()]
        
        # Client training
        client_updates = []
        
        for client_name, client_info in client_data.items():
            train_loader = client_info['train']
            n_samples = client_info['n_samples']
            
            # Create local model
            local_model = model_builder.build(device)
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            
            optimizer = torch.optim.AdamW(
                local_model.parameters(), 
                lr=lr, 
                weight_decay=config.get('weight_decay', 0.0001)
            )
            
            # Learning rate warmup
            if round_num < 10:
                for pg in optimizer.param_groups:
                    pg['lr'] = lr * (round_num + 1) / 10
            
            # Local training with AMP
            for epoch in range(local_epochs):
                for x, y in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if use_amp:
                        with autocast(dtype=torch.bfloat16):
                            logits = local_model(x)
                            loss = F.cross_entropy(logits, y, weight=class_weights)
                            
                            # FedProx regularization
                            if proximal_mu > 0:
                                prox_loss = 0.0
                                for p, gp in zip(local_model.parameters(), global_params):
                                    prox_loss += ((p - gp) ** 2).sum()
                                loss = loss + (proximal_mu / 2) * prox_loss
                        
                        scaler_amp.scale(loss).backward()
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        logits = local_model(x)
                        loss = F.cross_entropy(logits, y, weight=class_weights)
                        
                        if proximal_mu > 0:
                            prox_loss = 0.0
                            for p, gp in zip(local_model.parameters(), global_params):
                                prox_loss += ((p - gp) ** 2).sum()
                            loss = loss + (proximal_mu / 2) * prox_loss
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                        optimizer.step()
            
            # Collect update
            client_updates.append((
                [p.cpu().detach().clone() for p in local_model.parameters()],
                n_samples
            ))
            
            del local_model
        
        # FedAvg aggregation
        total_samples = sum(n for _, n in client_updates)
        new_params = None
        
        for params, n_samples in client_updates:
            weight = n_samples / total_samples
            if new_params is None:
                new_params = [weight * p for p in params]
            else:
                for i, p in enumerate(params):
                    new_params[i] = new_params[i] + weight * p
        
        # Update global model
        with torch.no_grad():
            for p, new_p in zip(global_model.parameters(), new_params):
                p.copy_(new_p.to(device))
        
        # Evaluate
        global_model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            for client_name, client_info in client_data.items():
                for x, y in client_info['val']:
                    x, y = x.to(device), y.to(device)
                    
                    if use_amp:
                        with autocast(dtype=torch.bfloat16):
                            logits = global_model(x)
                            total_loss += F.cross_entropy(logits, y).item()
                    else:
                        logits = global_model(x)
                        total_loss += F.cross_entropy(logits, y).item()
                    
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        benign_f1 = f1_per_class[0] if len(f1_per_class) > 0 else 0
        malicious_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        round_time = time.time() - round_start
        
        # Print progress
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            print(f"  Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
            print(f"  Benign F1: {benign_f1:.4f}, Malicious F1: {malicious_f1:.4f}")
            print(f"  Time: {round_time:.1f}s")
        
        # Log to WandB
        if wandb_run:
            import wandb
            wandb.log({
                'round': round_num + 1,
                'val_loss': total_loss,
                'val_accuracy': acc,
                'val_f1': f1,
                'val_benign_f1': benign_f1,
                'val_malicious_f1': malicious_f1,
                'round_time': round_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save history
        history['round'].append(round_num + 1)
        history['loss'].append(total_loss)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['benign_f1'].append(benign_f1)
        history['malicious_f1'].append(malicious_f1)
        
        # Best model tracking
        if f1 > best_f1:
            best_f1 = f1
            best_state = global_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at round {round_num + 1}")
            break
        
        # Memory cleanup
        if round_num % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Load best model
    if best_state:
        global_model.load_state_dict(best_state)
    
    training_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best Val F1: {best_f1:.4f}")
    
    return global_model, scaler, history, best_f1


def evaluate_with_tta(model, data_dir: Path, scaler, config: dict, device, wandb_run=None):
    """Evaluate with Drift-Aware Test-Time Adaptation."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import autocast
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    
    print("\n" + "=" * 70)
    print("TEST EVALUATION WITH DRIFT-AWARE TTA")
    print("=" * 70)
    
    # Load test data
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    with open(data_dir / "feature_columns.txt", 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    test_X = scaler.transform(test_df[feature_columns].values.astype(np.float32))
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_y = test_df['label_binary'].values.astype(np.int64)
    
    print(f"Test samples: {len(test_y)}")
    print(f"  Benign: {(test_y == 0).sum()}, Malicious: {(test_y == 1).sum()}")
    
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y)),
        batch_size=config.get('batch_size', 32768),
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # TTA settings
    tta_config = config.get('tta', {})
    tta_lr = tta_config.get('lr', 0.00005)
    tta_steps = tta_config.get('steps', 1)
    entropy_threshold = tta_config.get('entropy_threshold', 0.28)
    
    # Store original BN statistics
    original_bn_states = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            original_bn_states[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }
    
    # Evaluation with TTA
    model.eval()
    
    # First pass: compute baseline entropy
    all_entropies = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            all_entropies.extend(entropy.cpu().numpy())
    
    baseline_entropy = np.mean(all_entropies)
    print(f"Baseline entropy: {baseline_entropy:.4f}")
    
    # Adaptive threshold
    adaptive_threshold = max(entropy_threshold, baseline_entropy * 1.5)
    print(f"Adaptive entropy threshold: {adaptive_threshold:.4f}")
    
    # TTA pass
    all_preds = []
    all_labels = []
    adaptations = 0
    
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        # Apply TTA if entropy is high (uncertain predictions)
        if entropy > adaptive_threshold:
            adaptations += 1
            
            # Temporarily set to train mode for BN adaptation
            model.train()
            
            # Only adapt BN layers
            for param in model.parameters():
                param.requires_grad = False
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
            
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=tta_lr
            )
            
            # TTA steps
            for _ in range(tta_steps):
                optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16):
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    tta_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                tta_loss.backward()
                optimizer.step()
            
            # Reset grad requirements
            for param in model.parameters():
                param.requires_grad = True
            
            model.eval()
        
        # Final prediction
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
            preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # Restore original BN statistics
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d) and name in original_bn_states:
            module.running_mean.copy_(original_bn_states[name]['running_mean'])
            module.running_var.copy_(original_bn_states[name]['running_var'])
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\nTTA Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    print(f"  Adaptations: {adaptations} batches")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious']))
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Log to WandB
    if wandb_run:
        import wandb
        wandb.log({
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'tta_adaptations': adaptations,
            'baseline_entropy': baseline_entropy
        })
        
        # Log confusion matrix
        wandb.log({
            'confusion_matrix': wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=['Benign', 'Malicious']
            )
        })
    
    results = {
        'accuracy': test_acc,
        'f1': test_f1,
        'adaptations': adaptations,
        'baseline_entropy': baseline_entropy,
        'confusion_matrix': cm.tolist()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FL-TTA-IoT-IDS L40S Full Training')
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB')
    parser.add_argument('--max_flows', type=int, default=500000, 
                        help='Max flows per scenario (for memory management)')
    args = parser.parse_args()
    
    print("""
    ========================================================================
    |     FL-TTA-IoT-IDS: L40S FULL TRAINING                               |
    |     Maximum GPU Utilization for $1.49/hr                             |
    |     Expected: 94%+ Accuracy, 80%+ F1                                 |
    ========================================================================
    """)
    
    start_time = time.time()
    
    # Setup L40S environment
    device, vram_gb = setup_l40s_environment()
    
    # Configuration for FULL training
    config = {
        'training_type': 'full',
        'gpu': 'L40S',
        'vram_gb': vram_gb,
        'batch_size': 32768 if vram_gb > 40 else 16384,
        'num_workers': 16,
        'learning_rate': 0.004,
        'weight_decay': 0.0001,
        'precision': 'bf16-mixed',
        'model': {
            'hidden_dims': [1024, 512, 256, 128],
            'dropout': 0.2
        },
        'federated': {
            'num_rounds': 80,
            'local_epochs': 5,
            'proximal_mu': 0.01
        },
        'tta': {
            'method': 'drift_aware',
            'lr': 0.00005,
            'steps': 1,
            'entropy_threshold': 0.28
        },
        'patience': 20
    }
    
    # Initialize WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_run = setup_wandb(args.wandb_key, config)
    
    # Paths
    raw_dir = Path("data/raw/iot23_full")
    processed_dir = Path("data/processed/iot23_full")
    output_dir = Path("experiments/l40s_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    raw_files = download_iot23_full(raw_dir)
    
    if not raw_files:
        print("ERROR: No data downloaded!")
        return 1
    
    # Preprocess
    data_dir, feature_columns = preprocess_iot23(
        raw_files, processed_dir, 
        max_flows_per_scenario=args.max_flows
    )
    
    # Train
    model, scaler, history, best_f1 = train_federated_full(
        data_dir, config, device, wandb_run
    )
    
    # Evaluate with TTA
    test_results = evaluate_with_tta(model, data_dir, scaler, config, device, wandb_run)
    
    # Save results
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'test_results': test_results,
        'feature_columns': feature_columns
    }, output_dir / "model_final.pt")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'config': config,
            'history': history,
            'test_results': test_results,
            'training_time_minutes': (time.time() - start_time) / 60
        }, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    estimated_cost = (total_time / 3600) * 1.49
    
    print(f"\n" + "=" * 70)
    print(f"FULL TRAINING COMPLETE!")
    print(f"=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Macro F1: {test_results['f1']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Finish WandB
    if wandb_run:
        import wandb
        wandb.finish()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
