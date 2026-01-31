"""
Quick Test Script - Verifies pipeline works with synthetic tiny data
Run this BEFORE downloading real data to ensure everything is set up correctly

Usage:
    python test_pipeline.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def create_synthetic_iot_data(output_dir: str, num_samples: int = 10000, num_clients: int = 3):
    """Create synthetic IoT-like data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Feature columns (mimics IoT-23 features)
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'id.orig_p', 'id.resp_p',
        'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_orig', 'bytes_per_pkt_resp',
        'duration_log', 'total_bytes', 'total_pkts', 'is_well_known_port',
        'is_high_port', 'history_len',
        'proto_encoded', 'service_encoded', 'conn_state_encoded'
    ]
    
    # Save feature columns
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Generate data
    def generate_samples(n, client_shift=0):
        data = {}
        for col in feature_columns:
            if 'encoded' in col:
                data[col] = np.random.randint(0, 10, n)
            elif col in ['is_well_known_port', 'is_high_port']:
                data[col] = np.random.randint(0, 2, n)
            else:
                # Add client-specific shift to simulate non-IID
                data[col] = np.random.randn(n) + client_shift * 0.5
        
        # Binary labels (with some correlation to features)
        logits = (data['duration'] + data['orig_bytes'] * 0.5 + 
                  data['total_bytes'] * 0.3 + np.random.randn(n) * 0.5)
        data['label_binary'] = (logits > np.median(logits)).astype(int)
        
        # Timestamp for time-ordered splits
        data['ts'] = np.sort(np.random.uniform(0, 1000, n))
        data['scenario'] = 'test_scenario'
        
        return pd.DataFrame(data)
    
    # Create combined data
    all_dfs = []
    for client_id in range(num_clients):
        df = generate_samples(num_samples // num_clients, client_shift=client_id)
        df['scenario'] = f'client_{client_id}'
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Time-ordered split
    combined = combined.sort_values('ts').reset_index(drop=True)
    n = len(combined)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    combined.iloc[:train_end].to_parquet(output_dir / "train.parquet")
    combined.iloc[train_end:val_end].to_parquet(output_dir / "val.parquet")
    combined.iloc[val_end:].to_parquet(output_dir / "test.parquet")
    
    print(f"Created combined splits: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    
    # Create client splits
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(exist_ok=True)
    
    for client_id in range(num_clients):
        client_name = f'client_{client_id}'
        client_df = combined[combined['scenario'] == client_name].copy()
        client_df = client_df.sort_values('ts').reset_index(drop=True)
        
        cn = len(client_df)
        ct_end = int(cn * 0.7)
        cv_end = int(cn * 0.85)
        
        client_dir = clients_dir / client_name
        client_dir.mkdir(exist_ok=True)
        
        client_df.iloc[:ct_end].to_parquet(client_dir / "train.parquet")
        client_df.iloc[ct_end:cv_end].to_parquet(client_dir / "val.parquet")
        client_df.iloc[cv_end:].to_parquet(client_dir / "test.parquet")
        
        print(f"  {client_name}: train={ct_end}, val={cv_end-ct_end}, test={cn-cv_end}")
    
    print(f"\nSynthetic data saved to: {output_dir}")
    return output_dir


def test_datamodule():
    """Test data loading."""
    print("\n" + "="*60)
    print("Testing DataModule...")
    print("="*60)
    
    from src.data import IoT23DataModule
    
    dm = IoT23DataModule(
        data_dir="data/test_synthetic",
        batch_size=64,
        num_workers=0  # Avoid multiprocessing issues in test
    )
    dm.setup()
    
    # Test train loader
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    print(f"Train batch: x={x.shape}, y={y.shape}")
    print(f"Num features: {dm.num_features}")
    print(f"Num classes: {dm.num_classes}")
    
    assert x.shape[1] == dm.num_features, "Feature dimension mismatch"
    print("✓ DataModule test passed!")


def test_model():
    """Test model forward pass."""
    print("\n" + "="*60)
    print("Testing Models...")
    print("="*60)
    
    from src.models import create_model, MLP, FTTransformer
    
    input_dim = 23
    num_classes = 2
    batch_size = 32
    
    # Test MLP
    mlp = create_model(input_dim, num_classes, architecture='mlp', hidden_dims=[64, 32])
    x = torch.randn(batch_size, input_dim)
    out = mlp(x)
    print(f"MLP output: {out.shape}")
    assert out.shape == (batch_size, num_classes), "MLP output shape mismatch"
    print("✓ MLP test passed!")
    
    # Test FT-Transformer
    ft = create_model(input_dim, num_classes, architecture='ft_transformer', 
                      embed_dim=32, num_heads=2, num_layers=2)
    out = ft(x)
    print(f"FT-Transformer output: {out.shape}")
    assert out.shape == (batch_size, num_classes), "FT-Transformer output shape mismatch"
    print("✓ FT-Transformer test passed!")


def test_tta():
    """Test TTA methods."""
    print("\n" + "="*60)
    print("Testing TTA Methods...")
    print("="*60)
    
    from src.models import create_model
    from src.tta import TENT, EATA, DriftAwareTTA
    
    model = create_model(23, 2, 'mlp', hidden_dims=[64, 32])
    x = torch.randn(32, 23)
    
    # Test TENT
    tent = TENT(model, lr=1e-4, steps=1)
    out = tent(x)
    print(f"TENT output: {out.shape}")
    print("✓ TENT test passed!")
    
    # Test EATA
    model2 = create_model(23, 2, 'mlp', hidden_dims=[64, 32])
    eata = EATA(model2, lr=1e-4, steps=1)
    out = eata(x)
    print(f"EATA output: {out.shape}")
    print("✓ EATA test passed!")


def test_federated():
    """Test federated learning simulation."""
    print("\n" + "="*60)
    print("Testing Federated Learning...")
    print("="*60)
    
    from src.models import create_model
    from src.fl import run_federated_simulation
    from src.data import IoT23Dataset
    from torch.utils.data import DataLoader
    
    # Load test data
    data_dir = Path("data/test_synthetic")
    with open(data_dir / "feature_columns.txt", 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    
    # Create client data
    clients_dir = data_dir / "clients"
    client_data = {}
    
    for client_dir in clients_dir.iterdir():
        if not client_dir.is_dir():
            continue
        
        train_ds = IoT23Dataset(client_dir / "train.parquet", feature_columns, 'label_binary', fit_scaler=True)
        val_ds = IoT23Dataset(client_dir / "val.parquet", feature_columns, 'label_binary', scaler=train_ds.scaler)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        client_data[client_dir.name] = (train_loader, val_loader)
    
    print(f"Loaded {len(client_data)} clients")
    
    # Create model
    model = create_model(len(feature_columns), 2, 'mlp', hidden_dims=[64, 32])
    
    # Run FL (just 2 rounds for testing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = run_federated_simulation(
        model=model,
        client_data=client_data,
        num_rounds=2,
        local_epochs=1,
        learning_rate=1e-3,
        device=device
    )
    
    print(f"Final val F1: {history['val_f1_macro'][-1]:.4f}")
    print("✓ Federated Learning test passed!")


def test_full_pipeline():
    """Test full training pipeline."""
    print("\n" + "="*60)
    print("Testing Full Training Pipeline...")
    print("="*60)
    
    import subprocess
    
    # Run centralized training for 1 epoch
    result = subprocess.run([
        sys.executable, "train.py",
        "--mode", "centralized",
        "--data_dir", "data/test_synthetic",
        "--config", "configs/test_tiny.yaml",
        "--max_epochs", "1",
        "--debug"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Centralized training failed")
    
    print("✓ Full pipeline test passed!")


def main():
    print("="*60)
    print("FL-TTA IoT IDS - Pipeline Verification")
    print("="*60)
    
    # Create synthetic data
    print("\n1. Creating synthetic test data...")
    create_synthetic_iot_data("data/test_synthetic", num_samples=3000, num_clients=3)
    
    # Run tests
    print("\n2. Running component tests...")
    
    try:
        test_datamodule()
        test_model()
        test_tta()
        test_federated()
        # test_full_pipeline()  # Uncomment for full test
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nPipeline is ready. Next steps:")
        print("1. Download real IoT-23 data:")
        print("   python scripts/download_iot23.py --subset small")
        print("2. Preprocess:")
        print("   python scripts/preprocess.py")
        print("3. Train on L40S:")
        print("   python train.py --mode federated_tta --tta tent --config configs/l40s_full.yaml")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
