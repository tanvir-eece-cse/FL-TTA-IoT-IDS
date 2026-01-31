#!/usr/bin/env python3
"""
=============================================================================
CPU EVALUATION SCRIPT (i5 8th Gen + 12GB RAM)
=============================================================================
Optimized for testing on local machine with limited resources.
Uses a small portion of real IoT-23 data for quick validation.

Usage:
    python evaluate_cpu.py
    
Expected runtime: 3-5 minutes
=============================================================================
"""
import os
import sys
import time
import json
from pathlib import Path

# Change to script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))


def download_tiny_iot23(output_dir: Path):
    """Download IoT-23 scenarios for CPU testing."""
    import urllib.request
    import ssl
    
    print("\n" + "="*60)
    print("DOWNLOADING IOT-23 SAMPLES FOR CPU TESTING")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download scenarios that have BOTH benign and malicious traffic
    # Using scenario 1-1 which is documented to have mixed traffic
    scenarios = {
        "CTU-IoT-Malware-Capture-1-1": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled",
    }
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    downloaded = []
    for name, url in scenarios.items():
        output_file = output_dir / f"{name}_conn.log.labeled"
        
        if output_file.exists():
            print(f"{name}: Already exists")
            downloaded.append(output_file)
            continue
        
        print(f"Downloading {name}...")
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as response:
                data = response.read()
                with open(output_file, 'wb') as f:
                    f.write(data)
            print(f"  Downloaded: {output_file.stat().st_size / 1024:.1f} KB")
            downloaded.append(output_file)
        except Exception as e:
            print(f"  Failed: {e}")
    
    return downloaded


def preprocess_tiny(raw_files: list, output_dir: Path):
    """Lightweight preprocessing for CPU."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("PREPROCESSING FOR CPU")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    columns = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
        'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'tunnel_parents', 'label', 'detailed-label'
    ]
    
    # Parse ALL downloaded files - IoT-23 Zeek format
    # Based on #fields header:
    # ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, 
    # orig_bytes, resp_bytes, conn_state, local_orig, local_resp, missed_bytes, history,
    # orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes, tunnel_parents, label, detailed-label
    for filepath in raw_files:
        print(f"Parsing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        rows = []
        for line in lines:
            # Skip headers and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            
            # IoT-23 specific: the last field contains "tunnel_parents   label   detailed-label"
            # separated by multiple spaces (not tabs)
            if len(parts) >= 21:  # IoT-23 data lines have 21 tab-separated fields
                # Parse the last column which contains 3 space-separated values
                last_col = parts[-1].strip()
                last_parts = last_col.split()  # Split by whitespace
                
                # Extract label (second element if exists, otherwise check for Malicious/Benign)
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
                    'orig_pkts': parts[17],
                    'orig_ip_bytes': parts[18],
                    'resp_pkts': parts[19],
                    'resp_ip_bytes': parts[20] if len(parts) > 20 else '0',
                    'label': label,
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df['scenario'] = filepath.stem
        print(f"  Loaded {len(df)} flows")
        all_dfs.append(df)
    
    # Combine all files
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined flows: {len(df)}")
    
    # Debug: Show unique labels
    unique_labels = df['label'].unique()[:10]
    print(f"Sample labels: {unique_labels}")
    
    # Labels - IoT-23 uses "Benign" for benign traffic
    # Everything else (including "Malicious", "-", attack names) is malicious
    df['label_binary'] = df['label'].apply(
        lambda x: 0 if 'benign' in str(x).lower() else 1
    )
    benign_count = (df['label_binary'] == 0).sum()
    malicious_count = (df['label_binary'] == 1).sum()
    print(f"Before sampling - Benign: {benign_count}, Malicious: {malicious_count}")
    
    # Balance and sample for CPU speed (take equal samples from each class)
    benign_df = df[df['label_binary'] == 0]
    malicious_df = df[df['label_binary'] == 1]
    
    sample_per_class = min(1000, len(benign_df), len(malicious_df))
    
    if sample_per_class < 100:
        # If one class is too small, just take what we have
        sample_per_class = min(len(benign_df), len(malicious_df)) if min(len(benign_df), len(malicious_df)) > 0 else max(len(benign_df), len(malicious_df))
        print(f"Warning: Imbalanced data, using {sample_per_class} samples per class")
    
    if len(benign_df) > 0 and len(malicious_df) > 0:
        df = pd.concat([
            benign_df.sample(n=min(sample_per_class, len(benign_df)), random_state=42),
            malicious_df.sample(n=min(sample_per_class, len(malicious_df)), random_state=42)
        ], ignore_index=True)
    else:
        df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"After balancing: {len(df)} flows")
    print(f"  Benign: {(df['label_binary'] == 0).sum()}, Malicious: {(df['label_binary'] == 1).sum()}")
    
    # Simple numeric features
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
                    'id.orig_p', 'id.resp_p']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('-', '0'), errors='coerce').fillna(0)
    
    # Basic engineered features
    df['bytes_ratio'] = (df['orig_bytes'] + 1) / (df['resp_bytes'] + 1)
    df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
    df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']
    
    # Categorical
    for col in ['proto', 'service', 'conn_state']:
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(df[col].fillna('unknown').unique())}
            df[f'{col}_encoded'] = df[col].fillna('unknown').map(mapping)
    
    # Feature columns (small set for CPU)
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'id.orig_p', 'id.resp_p', 'bytes_ratio', 'total_bytes', 'total_pkts',
        'proto_encoded', 'service_encoded', 'conn_state_encoded'
    ]
    feature_columns = [c for c in feature_columns if c in df.columns]
    
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    cols_to_save = feature_columns + ['label_binary', 'scenario']
    
    df.iloc[:train_end][cols_to_save].to_parquet(output_dir / "train.parquet")
    df.iloc[train_end:val_end][cols_to_save].to_parquet(output_dir / "val.parquet")
    df.iloc[val_end:][cols_to_save].to_parquet(output_dir / "test.parquet")
    
    print(f"Saved: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    
    # Single client
    clients_dir = output_dir / "clients" / "scenario_8"
    clients_dir.mkdir(parents=True, exist_ok=True)
    
    df.iloc[:train_end][cols_to_save].to_parquet(clients_dir / "train.parquet")
    df.iloc[train_end:val_end][cols_to_save].to_parquet(clients_dir / "val.parquet")
    df.iloc[val_end:][cols_to_save].to_parquet(clients_dir / "test.parquet")
    
    return output_dir, feature_columns


def train_and_evaluate_cpu(data_dir: Path, feature_columns: list):
    """Simple training and evaluation on CPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    print("\n" + "="*60)
    print("TRAINING ON CPU (i5 8th Gen Optimized)")
    print("="*60)
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare data
    scaler = StandardScaler()
    
    train_X = scaler.fit_transform(train_df[feature_columns].values.astype(np.float32))
    train_y = train_df['label_binary'].values.astype(np.int64)
    
    val_X = scaler.transform(val_df[feature_columns].values.astype(np.float32))
    val_y = val_df['label_binary'].values.astype(np.int64)
    
    test_X = scaler.transform(test_df[feature_columns].values.astype(np.float32))
    test_y = test_df['label_binary'].values.astype(np.int64)
    
    # Replace NaN
    train_X = np.nan_to_num(train_X, 0)
    val_X = np.nan_to_num(val_X, 0)
    test_X = np.nan_to_num(test_X, 0)
    
    # Small batch for CPU
    batch_size = 32
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y)),
        batch_size=batch_size
    )
    
    # Simple MLP for CPU
    input_dim = len(feature_columns)
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    epochs = 20
    best_val_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).numpy()
                val_preds.extend(preds)
                val_labels.extend(y.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Test evaluation
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).numpy()
            test_preds.extend(preds)
    
    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds, average='macro', zero_division=0)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Only print report if we have both classes
    unique_labels = np.unique(test_y)
    if len(unique_labels) == 2:
        print(f"\nClassification Report:")
        print(classification_report(test_y, test_preds, target_names=['Benign', 'Malicious']))
    else:
        print(f"\nNote: Test set only has class {unique_labels[0]}")
    
    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'best_val_f1': best_val_f1
    }


def main():
    print("""
    ================================================================
    |     FL-TTA-IoT-IDS: CPU EVALUATION                           |
    |     Optimized for i5 8th Gen + 12GB RAM                      |
    |     Expected runtime: 3-5 minutes                            |
    ================================================================
    """)
    
    start_time = time.time()
    
    # Paths
    raw_dir = Path("data/raw/iot23_cpu_test")
    processed_dir = Path("data/processed/iot23_cpu_test")
    
    # Download
    raw_files = download_tiny_iot23(raw_dir)
    
    if not raw_files:
        print("ERROR: Could not download data!")
        return 1
    
    # Preprocess
    data_dir, feature_columns = preprocess_tiny(raw_files, processed_dir)
    
    # Train and evaluate
    results = train_and_evaluate_cpu(data_dir, feature_columns)
    
    # Save results
    output_dir = Path("experiments/cpu_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'results': results,
            'runtime_seconds': time.time() - start_time
        }, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"CPU EVALUATION COMPLETE!")
    print(f"="*60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Test F1: {results['f1']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
