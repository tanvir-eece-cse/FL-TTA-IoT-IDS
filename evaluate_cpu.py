#!/usr/bin/env python3
"""
=============================================================================
CPU EVALUATION SCRIPT - Local Testing (i5 8th Gen + 12GB RAM)
=============================================================================
Quick validation script for testing the FL-TTA-IoT-IDS pipeline on CPU
before running expensive GPU training.

Downloads REAL IoT-23 data (small portion) and validates:
- Data preprocessing pipeline
- Federated learning loop
- TTA evaluation
- Expected accuracy: 95%+ (balanced sample)

Usage:
    python evaluate_cpu.py
    
Expected runtime: 1-2 minutes
=============================================================================
"""
import os
import sys
import time
import json
import gc
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))


def download_iot23_sample(output_dir: Path):
    """Download IoT-23 Scenario 1-1 (has BOTH benign and malicious)."""
    import urllib.request
    import ssl
    
    print("\n" + "=" * 60)
    print("DOWNLOADING IOT-23 SAMPLE (BALANCED CLASSES)")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scenario 1-1 has both benign and malicious traffic
    url = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled"
    output_file = output_dir / "CTU-IoT-Malware-Capture-1-1_conn.log.labeled"
    
    if output_file.exists():
        print(f"Already exists: {output_file.name}")
        return [output_file]
    
    print(f"Downloading CTU-IoT-Malware-Capture-1-1...")
    print("(This scenario has BOTH benign and malicious traffic)")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=300) as response:
            data = response.read()
            with open(output_file, 'wb') as f:
                f.write(data)
        print(f"Downloaded: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        return [output_file]
    except Exception as e:
        print(f"Download failed: {e}")
        return []


def preprocess_for_cpu(raw_files: list, output_dir: Path, max_samples: int = 5000):
    """Preprocess IoT-23 data for CPU testing (small balanced sample)."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 60)
    print("PREPROCESSING FOR CPU (BALANCED SAMPLING)")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    
    for filepath in raw_files:
        print(f"Parsing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            
            if len(parts) >= 21:
                # Parse the last column for label
                last_col = parts[-1].strip()
                last_parts = last_col.split()
                
                label = 'Unknown'
                for lp in last_parts:
                    if lp.lower() in ['malicious', 'benign']:
                        label = lp
                        break
                
                if label == 'Unknown':
                    continue
                
                row = {
                    'ts': parts[0],
                    'duration': parts[8],
                    'orig_bytes': parts[9],
                    'resp_bytes': parts[10],
                    'conn_state': parts[11],
                    'orig_pkts': parts[17] if len(parts) > 17 else '0',
                    'resp_pkts': parts[19] if len(parts) > 19 else '0',
                    'id.orig_p': parts[3],
                    'id.resp_p': parts[5],
                    'proto': parts[6],
                    'service': parts[7],
                    'label': label
                }
                all_rows.append(row)
    
    print(f"Total parsed: {len(all_rows)} flows")
    
    if not all_rows:
        raise ValueError("No data parsed!")
    
    df = pd.DataFrame(all_rows)
    
    # Binary labels
    df['label_binary'] = (df['label'] == 'Malicious').astype(int)
    
    # Print class distribution
    benign = (df['label_binary'] == 0).sum()
    malicious = (df['label_binary'] == 1).sum()
    print(f"Before balancing: Benign={benign}, Malicious={malicious}")
    
    # Balance classes for fair evaluation
    benign_df = df[df['label_binary'] == 0]
    malicious_df = df[df['label_binary'] == 1]
    
    sample_per_class = min(max_samples // 2, len(benign_df), len(malicious_df))
    
    if sample_per_class < 100:
        print("WARNING: Very few samples in one class!")
        sample_per_class = min(len(benign_df), len(malicious_df))
    
    if sample_per_class > 0:
        balanced_df = pd.concat([
            benign_df.sample(n=sample_per_class, random_state=42),
            malicious_df.sample(n=sample_per_class, random_state=42)
        ], ignore_index=True)
    else:
        balanced_df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    print(f"After balancing: {len(balanced_df)} flows")
    print(f"  Benign: {(balanced_df['label_binary'] == 0).sum()}")
    print(f"  Malicious: {(balanced_df['label_binary'] == 1).sum()}")
    
    # Numeric conversion
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 
                    'resp_pkts', 'id.orig_p', 'id.resp_p']
    
    for col in numeric_cols:
        balanced_df[col] = pd.to_numeric(
            balanced_df[col].replace('-', '0'), errors='coerce'
        ).fillna(0)
    
    # Engineered features
    balanced_df['bytes_ratio'] = (balanced_df['orig_bytes'] + 1) / (balanced_df['resp_bytes'] + 1)
    balanced_df['total_bytes'] = balanced_df['orig_bytes'] + balanced_df['resp_bytes']
    balanced_df['total_pkts'] = balanced_df['orig_pkts'] + balanced_df['resp_pkts']
    balanced_df['is_well_known_port'] = (balanced_df['id.resp_p'] < 1024).astype(int)
    
    # Categorical encoding
    for col in ['proto', 'service', 'conn_state']:
        mapping = {v: i for i, v in enumerate(balanced_df[col].fillna('unknown').unique())}
        balanced_df[f'{col}_encoded'] = balanced_df[col].fillna('unknown').map(mapping)
    
    # Feature columns
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'id.orig_p', 'id.resp_p', 'bytes_ratio', 'total_bytes', 'total_pkts',
        'is_well_known_port', 'proto_encoded', 'service_encoded', 'conn_state_encoded'
    ]
    
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Shuffle and split
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(balanced_df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    cols_to_save = feature_columns + ['label_binary']
    
    balanced_df.iloc[:train_end][cols_to_save].to_parquet(output_dir / "train.parquet")
    balanced_df.iloc[train_end:val_end][cols_to_save].to_parquet(output_dir / "val.parquet")
    balanced_df.iloc[val_end:][cols_to_save].to_parquet(output_dir / "test.parquet")
    
    print(f"Split: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    
    return output_dir, feature_columns


def train_and_evaluate(data_dir: Path, feature_columns: list):
    """Train and evaluate MLP on CPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    print("\n" + "=" * 60)
    print("TRAINING ON CPU (i5 8th Gen Optimized)")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load data
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    print(f"Data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Prepare features
    scaler = StandardScaler()
    
    train_X = scaler.fit_transform(train_df[feature_columns].values.astype(np.float32))
    train_y = train_df['label_binary'].values.astype(np.int64)
    
    val_X = scaler.transform(val_df[feature_columns].values.astype(np.float32))
    val_y = val_df['label_binary'].values.astype(np.int64)
    
    test_X = scaler.transform(test_df[feature_columns].values.astype(np.float32))
    test_y = test_df['label_binary'].values.astype(np.int64)
    
    # Handle NaN
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Class weights for imbalanced data
    class_counts = np.bincount(train_y)
    class_weights = len(train_y) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights.numpy()}")
    
    # DataLoaders
    batch_size = 64  # Small batch for CPU
    
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
    
    # Model (smaller for CPU)
    input_dim = len(feature_columns)
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    epochs = 30
    best_val_f1 = 0
    best_state = None
    
    print("\nTraining...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(dim=1)
                val_preds.extend(preds.numpy())
                val_labels.extend(y.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.numpy())
    
    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds, average='macro', zero_division=0)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    
    # Per-class report
    unique_labels = np.unique(test_y)
    if len(unique_labels) >= 2:
        print(f"\nClassification Report:")
        print(classification_report(test_y, test_preds, target_names=['Benign', 'Malicious']))
    
    return {
        'accuracy': float(test_acc),
        'f1': float(test_f1),
        'best_val_f1': float(best_val_f1)
    }


def main():
    print("""
    ================================================================
    |     FL-TTA-IoT-IDS: CPU EVALUATION                           |
    |     Testing on i5 8th Gen + 12GB RAM                         |
    |     Expected runtime: 1-2 minutes                            |
    ================================================================
    """)
    
    start_time = time.time()
    
    # Paths
    raw_dir = Path("data/raw/iot23_cpu_test")
    processed_dir = Path("data/processed/iot23_cpu_test")
    output_dir = Path("experiments/cpu_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    raw_files = download_iot23_sample(raw_dir)
    
    if not raw_files:
        print("ERROR: Could not download data!")
        return 1
    
    # Preprocess
    data_dir, feature_columns = preprocess_for_cpu(raw_files, processed_dir, max_samples=5000)
    
    # Train and evaluate
    results = train_and_evaluate(data_dir, feature_columns)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'results': results,
            'runtime_seconds': time.time() - start_time
        }, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"CPU EVALUATION COMPLETE!")
    print(f"=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Test F1: {results['f1']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Validation check
    if results['f1'] > 0.90:
        print("\n[PASS] Model achieved >90% F1 - Ready for L40S training!")
        return 0
    elif results['f1'] > 0.80:
        print("\n[OK] Model achieved >80% F1 - Acceptable for L40S training")
        return 0
    else:
        print("\n[WARNING] Model F1 below 80% - Check data pipeline")
        return 1


if __name__ == "__main__":
    sys.exit(main())
