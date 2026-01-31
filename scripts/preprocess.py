"""
IoT-23 Preprocessing Pipeline
Converts Zeek conn.log.labeled files to processed parquet with:
- Feature engineering
- Label encoding (binary + multiclass)
- Time-ordered splits for drift evaluation
- Client-based splits for federated learning
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Zeek conn.log column names
ZEEK_COLUMNS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
    'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents',
    'label', 'detailed-label'
]

# Features to use for ML
NUMERIC_FEATURES = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'id.orig_p', 'id.resp_p'
]

CATEGORICAL_FEATURES = ['proto', 'service', 'conn_state']

# Label mappings
BINARY_LABEL_MAP = {
    'Benign': 0,
    'benign': 0,
    'Malicious': 1,
    'malicious': 1
}

# Attack type groupings for multiclass
ATTACK_GROUPS = {
    'Benign': 'Benign',
    'benign': 'Benign',
    'PartOfAHorizontalPortScan': 'PortScan',
    'DDoS': 'DDoS',
    'C&C': 'C&C',
    'C&C-HeartBeat': 'C&C',
    'C&C-Torii': 'C&C',
    'C&C-HeartBeat-Attack': 'C&C',
    'C&C-FileDownload': 'C&C',
    'C&C-Mirai': 'C&C',
    'Attack': 'Attack',
    'Okiru': 'Malware',
    'Okiru-Attack': 'Malware',
    'Torii': 'Malware',
    'Hajime': 'Malware',
    'Mirai': 'Malware',
    'FileDownload': 'Malware',
}


def parse_zeek_log(filepath: Path) -> pd.DataFrame:
    """Parse a Zeek conn.log.labeled file."""
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= len(ZEEK_COLUMNS):
                rows.append(parts[:len(ZEEK_COLUMNS)])
            elif len(parts) >= 21:  # Minimum required columns
                # Pad with empty strings for missing columns
                parts.extend([''] * (len(ZEEK_COLUMNS) - len(parts)))
                rows.append(parts)
    
    df = pd.DataFrame(rows, columns=ZEEK_COLUMNS)
    return df


def clean_and_engineer_features(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Clean data and engineer features."""
    df = df.copy()
    
    # Add scenario identifier for FL client splits
    df['scenario'] = scenario_name
    
    # Convert timestamp
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    
    # Convert numeric features
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col].replace('-', '0'), errors='coerce').fillna(0)
    
    # Handle categorical features
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].replace('-', 'unknown').fillna('unknown')
    
    # Binary label
    df['label_binary'] = df['label'].map(lambda x: 0 if 'benign' in str(x).lower() else 1)
    
    # Multiclass label (attack type groups)
    df['detailed-label'] = df['detailed-label'].replace('-', 'Unknown').fillna('Unknown')
    df['label_multi'] = df['detailed-label'].map(
        lambda x: ATTACK_GROUPS.get(x.split('-')[0] if '-' in str(x) else x, 'Other')
    )
    
    # Feature engineering
    df['bytes_ratio'] = (df['orig_bytes'] + 1) / (df['resp_bytes'] + 1)
    df['pkts_ratio'] = (df['orig_pkts'] + 1) / (df['resp_pkts'] + 1)
    df['bytes_per_pkt_orig'] = df['orig_bytes'] / (df['orig_pkts'] + 1)
    df['bytes_per_pkt_resp'] = df['resp_bytes'] / (df['resp_pkts'] + 1)
    df['duration_log'] = np.log1p(df['duration'])
    df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
    df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']
    
    # Port-based features
    df['is_well_known_port'] = (df['id.resp_p'] < 1024).astype(int)
    df['is_high_port'] = (df['id.resp_p'] > 49152).astype(int)
    
    # History-based features (connection behavior)
    df['history_len'] = df['history'].apply(lambda x: len(str(x)) if x != '-' else 0)
    
    return df


def encode_categoricals(df: pd.DataFrame, encoders: Optional[Dict] = None, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical features using label encoding."""
    if encoders is None:
        encoders = {}
    
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            # Add 'unknown' to handle unseen categories
            all_values = list(df[col].unique()) + ['unknown']
            le.fit(all_values)
            encoders[col] = le
        
        # Transform with handling for unseen values
        df[col + '_encoded'] = df[col].apply(
            lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ 
            else encoders[col].transform(['unknown'])[0]
        )
    
    return df, encoders


def create_time_ordered_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-ordered train/val/test splits for drift evaluation."""
    df = df.sort_values('ts').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


def create_client_splits(df: pd.DataFrame, n_clients: int = None) -> Dict[str, pd.DataFrame]:
    """Create client-based splits for federated learning (each scenario = 1 client)."""
    scenarios = df['scenario'].unique()
    if n_clients is not None and n_clients < len(scenarios):
        scenarios = np.random.choice(scenarios, n_clients, replace=False)
    
    return {scenario: df[df['scenario'] == scenario] for scenario in scenarios}


def get_feature_columns():
    """Get list of feature columns for ML."""
    engineered = [
        'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_orig', 'bytes_per_pkt_resp',
        'duration_log', 'total_bytes', 'total_pkts', 'is_well_known_port',
        'is_high_port', 'history_len'
    ]
    encoded_cats = [col + '_encoded' for col in CATEGORICAL_FEATURES]
    return NUMERIC_FEATURES + engineered + encoded_cats


def main():
    parser = argparse.ArgumentParser(description="Preprocess IoT-23 Dataset")
    parser.add_argument("--input_dir", type=str, default="data/raw/iot23",
                        help="Input directory with raw Zeek logs")
    parser.add_argument("--output_dir", type=str, default="data/processed/iot23",
                        help="Output directory for processed data")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--sample_per_scenario", type=int, default=None,
                        help="Max samples per scenario (for testing)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all scenario directories
    scenarios = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(scenarios)} scenarios")
    
    all_dfs = []
    
    for scenario_dir in tqdm(scenarios, desc="Processing scenarios"):
        log_file = scenario_dir / "conn.log.labeled"
        if not log_file.exists():
            print(f"Skipping {scenario_dir.name}: no conn.log.labeled found")
            continue
        
        print(f"\nProcessing {scenario_dir.name}...")
        
        # Parse
        df = parse_zeek_log(log_file)
        print(f"  Parsed {len(df)} rows")
        
        # Sample if requested (for quick testing)
        if args.sample_per_scenario and len(df) > args.sample_per_scenario:
            df = df.sample(n=args.sample_per_scenario, random_state=42)
            print(f"  Sampled to {len(df)} rows")
        
        # Clean and engineer features
        df = clean_and_engineer_features(df, scenario_dir.name)
        
        all_dfs.append(df)
    
    # Combine all scenarios
    print("\nCombining all scenarios...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    
    # Encode categoricals (fit on all data)
    print("Encoding categorical features...")
    combined_df, encoders = encode_categoricals(combined_df, fit=True)
    
    # Save encoders
    joblib.dump(encoders, output_dir / "encoders.pkl")
    
    # Print label distribution
    print("\nLabel distribution (binary):")
    print(combined_df['label_binary'].value_counts())
    print("\nLabel distribution (multiclass):")
    print(combined_df['label_multi'].value_counts())
    
    # Create and save time-ordered splits
    print("\nCreating time-ordered splits...")
    train_df, val_df, test_df = create_time_ordered_splits(
        combined_df, args.train_ratio, args.val_ratio
    )
    
    # Save splits
    feature_cols = get_feature_columns()
    
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df.to_parquet(output_dir / f"{name}.parquet", index=False)
        print(f"  {name}: {len(split_df)} samples")
    
    # Save feature column list
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # Create client-based splits for FL
    print("\nCreating client-based splits for federated learning...")
    client_dir = output_dir / "clients"
    client_dir.mkdir(exist_ok=True)
    
    client_splits = create_client_splits(combined_df)
    for client_name, client_df in client_splits.items():
        # Time-ordered split within each client
        c_train, c_val, c_test = create_time_ordered_splits(client_df, args.train_ratio, args.val_ratio)
        
        client_out = client_dir / client_name
        client_out.mkdir(exist_ok=True)
        
        c_train.to_parquet(client_out / "train.parquet", index=False)
        c_val.to_parquet(client_out / "val.parquet", index=False)
        c_test.to_parquet(client_out / "test.parquet", index=False)
        
        print(f"  {client_name}: {len(c_train)}/{len(c_val)}/{len(c_test)} (train/val/test)")
    
    print(f"\nPreprocessing complete! Data saved to: {output_dir}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Clients (scenarios): {len(client_splits)}")


if __name__ == "__main__":
    main()
