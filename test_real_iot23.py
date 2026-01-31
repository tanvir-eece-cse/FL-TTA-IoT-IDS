"""
Test REAL IoT-23 Data Training (NO SYNTHETIC DATA)
===================================================
Downloads actual malware traffic from Stratosphere IPS IoT-23 dataset.

Usage:
    python test_real_iot23.py
"""
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import ssl

# Correct URLs for IoT-23 Individual Scenarios
# Using small scenarios for quick testing
IOT23_SCENARIOS = {
    # CTU-IoT-Malware-Capture-44-1 (Mirai) - SMALLEST: only 237 flows
    "CTU-IoT-Malware-Capture-44-1": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-44-1/bro/conn.log.labeled",
        "labels": {"Benign": 211, "C&C": 14, "C&C-FileDownload": 11, "DDoS": 1}
    },
    # CTU-IoT-Malware-Capture-8-1 (Hakai) - Small: ~10K flows
    "CTU-IoT-Malware-Capture-8-1": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled",
        "labels": {"Benign": 2181, "C&C": 8222}
    },
    # CTU-IoT-Malware-Capture-34-1 (Mirai) - Small: ~23K flows  
    "CTU-IoT-Malware-Capture-34-1": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled",
        "labels": {"Benign": 1923, "C&C": 6706, "DDoS": 14394, "PartOfAHorizontalPortScan": 122}
    },
}


def download_real_iot23(output_dir: Path, max_scenarios: int = 2):
    """Download REAL IoT-23 labeled conn.log files."""
    print("\n" + "="*60)
    print("DOWNLOADING REAL IOT-23 MALWARE TRAFFIC DATA")
    print("Source: Stratosphere IPS, CTU University, Czech Republic")
    print("="*60 + "\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SSL context that doesn't verify (some servers have cert issues)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    downloaded_files = []
    
    for i, (scenario_name, info) in enumerate(IOT23_SCENARIOS.items()):
        if i >= max_scenarios:
            break
            
        url = info["url"]
        output_file = output_dir / f"{scenario_name}_conn.log.labeled"
        
        print(f"[{i+1}/{max_scenarios}] Downloading {scenario_name}...")
        print(f"    URL: {url}")
        
        try:
            # Try with SSL context
            with urllib.request.urlopen(url, context=ctx, timeout=60) as response:
                data = response.read()
                with open(output_file, 'wb') as f:
                    f.write(data)
            
            file_size = output_file.stat().st_size / 1024
            print(f"    Downloaded: {file_size:.1f} KB")
            downloaded_files.append(output_file)
            
        except Exception as e:
            print(f"    ERROR: {e}")
            # Try alternative URL format
            alt_url = url.replace("/bro/", "/")
            print(f"    Trying alternative: {alt_url}")
            try:
                with urllib.request.urlopen(alt_url, context=ctx, timeout=60) as response:
                    data = response.read()
                    with open(output_file, 'wb') as f:
                        f.write(data)
                file_size = output_file.stat().st_size / 1024
                print(f"    Downloaded: {file_size:.1f} KB")
                downloaded_files.append(output_file)
            except Exception as e2:
                print(f"    FAILED: {e2}")
    
    return downloaded_files


def parse_zeek_conn_log(filepath: Path):
    """Parse Zeek conn.log.labeled file into DataFrame."""
    import pandas as pd
    
    print(f"\nParsing: {filepath.name}")
    
    # Read the file and find the header
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find header line (starts with #fields)
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#fields'):
            header_line = line.strip().replace('#fields\t', '').split('\t')
            data_start = i + 1
        elif line.startswith('#types'):
            data_start = i + 1
        elif not line.startswith('#') and header_line:
            data_start = i
            break
    
    if header_line is None:
        # Default Zeek conn.log columns + label columns
        header_line = [
            'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
            'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
            'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
            'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'tunnel_parents', 'label', 'detailed-label'
        ]
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                data_start = i
                break
    
    # Parse data lines
    data_rows = []
    for line in lines[data_start:]:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) >= len(header_line):
            data_rows.append(parts[:len(header_line)])
        elif len(parts) > 5:  # Has at least some data
            # Pad with empty strings
            parts.extend([''] * (len(header_line) - len(parts)))
            data_rows.append(parts)
    
    if not data_rows:
        print(f"  WARNING: No data rows found in {filepath}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows, columns=header_line)
    print(f"  Loaded {len(df)} flows")
    
    return df


def preprocess_iot23_data(raw_files: list, output_dir: Path):
    """Preprocess IoT-23 data for training."""
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("PREPROCESSING REAL IOT-23 DATA")
    print("="*60 + "\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    for filepath in raw_files:
        df = parse_zeek_conn_log(filepath)
        if len(df) > 0:
            df['scenario'] = filepath.stem.replace('_conn.log.labeled', '')
            all_dfs.append(df)
    
    if not all_dfs:
        print("ERROR: No data loaded!")
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal flows: {len(combined)}")
    
    # Extract label column (try both formats)
    if 'label' in combined.columns:
        label_col = 'label'
    elif 'detailed-label' in combined.columns:
        label_col = 'detailed-label'
    else:
        # Find column containing labels
        for col in combined.columns:
            if combined[col].astype(str).str.contains('Benign|Malicious|C&C|DDoS', case=False, na=False).any():
                label_col = col
                break
        else:
            print("ERROR: Could not find label column!")
            print(f"Columns: {combined.columns.tolist()}")
            return None
    
    print(f"\nUsing label column: {label_col}")
    print(f"Label distribution:")
    label_counts = combined[label_col].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Create binary labels (Benign vs Malicious)
    combined['label_binary'] = (~combined[label_col].str.contains('Benign', case=False, na=False)).astype(int)
    combined['label_multi'] = combined[label_col]
    
    print(f"\nBinary labels: Benign={sum(combined['label_binary']==0)}, Malicious={sum(combined['label_binary']==1)}")
    
    # Select and engineer features
    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col].replace('-', '0'), errors='coerce').fillna(0)
    
    # Add port columns
    if 'id.orig_p' in combined.columns:
        combined['id.orig_p'] = pd.to_numeric(combined['id.orig_p'].replace('-', '0'), errors='coerce').fillna(0)
    if 'id.resp_p' in combined.columns:
        combined['id.resp_p'] = pd.to_numeric(combined['id.resp_p'].replace('-', '0'), errors='coerce').fillna(0)
    
    # Engineer features
    combined['bytes_ratio'] = (combined['orig_bytes'] + 1) / (combined['resp_bytes'] + 1)
    combined['pkts_ratio'] = (combined['orig_pkts'] + 1) / (combined['resp_pkts'] + 1)
    combined['bytes_per_pkt_orig'] = combined['orig_bytes'] / (combined['orig_pkts'] + 1)
    combined['bytes_per_pkt_resp'] = combined['resp_bytes'] / (combined['resp_pkts'] + 1)
    combined['duration_log'] = np.log1p(combined['duration'])
    combined['total_bytes'] = combined['orig_bytes'] + combined['resp_bytes']
    combined['total_pkts'] = combined['orig_pkts'] + combined['resp_pkts']
    
    if 'id.resp_p' in combined.columns:
        combined['is_well_known_port'] = (combined['id.resp_p'] < 1024).astype(int)
        combined['is_high_port'] = (combined['id.resp_p'] > 49152).astype(int)
    else:
        combined['is_well_known_port'] = 0
        combined['is_high_port'] = 0
    
    if 'history' in combined.columns:
        combined['history_len'] = combined['history'].fillna('').str.len()
    else:
        combined['history_len'] = 0
    
    # Encode categorical - use simple integer encoding instead of sklearn
    if 'proto' in combined.columns:
        proto_map = {v: i for i, v in enumerate(combined['proto'].fillna('unknown').unique())}
        combined['proto_encoded'] = combined['proto'].fillna('unknown').map(proto_map)
    else:
        combined['proto_encoded'] = 0
        
    if 'service' in combined.columns:
        service_map = {v: i for i, v in enumerate(combined['service'].fillna('-').unique())}
        combined['service_encoded'] = combined['service'].fillna('-').map(service_map)
    else:
        combined['service_encoded'] = 0
        
    if 'conn_state' in combined.columns:
        conn_map = {v: i for i, v in enumerate(combined['conn_state'].fillna('OTH').unique())}
        combined['conn_state_encoded'] = combined['conn_state'].fillna('OTH').map(conn_map)
    else:
        combined['conn_state_encoded'] = 0
    
    # Define feature columns
    feature_columns = [
        'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
        'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
        'bytes_ratio', 'pkts_ratio', 'bytes_per_pkt_orig', 'bytes_per_pkt_resp',
        'duration_log', 'total_bytes', 'total_pkts', 'is_well_known_port',
        'is_high_port', 'history_len', 'proto_encoded', 'service_encoded', 
        'conn_state_encoded'
    ]
    
    # Add port features if available
    if 'id.orig_p' in combined.columns:
        feature_columns.extend(['id.orig_p', 'id.resp_p'])
    
    # Filter to available columns
    feature_columns = [c for c in feature_columns if c in combined.columns]
    
    # Save feature columns
    with open(output_dir / "feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print(f"\nFeatures: {len(feature_columns)}")
    
    # Shuffle and split
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(combined)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = combined.iloc[:train_end]
    val_df = combined.iloc[train_end:val_end]
    test_df = combined.iloc[val_end:]
    
    # Save splits
    cols_to_save = feature_columns + ['label_binary', 'label_multi', 'scenario']
    train_df[cols_to_save].to_parquet(output_dir / "train.parquet")
    val_df[cols_to_save].to_parquet(output_dir / "val.parquet")
    test_df[cols_to_save].to_parquet(output_dir / "test.parquet")
    
    print(f"\nSaved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create client splits based on scenarios
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(exist_ok=True)
    
    scenarios = combined['scenario'].unique()
    for scenario in scenarios:
        scenario_df = combined[combined['scenario'] == scenario].copy()
        sn = len(scenario_df)
        st_end = int(sn * 0.7)
        sv_end = int(sn * 0.85)
        
        client_dir = clients_dir / scenario.replace('-', '_')
        client_dir.mkdir(exist_ok=True)
        
        scenario_df.iloc[:st_end][cols_to_save].to_parquet(client_dir / "train.parquet")
        scenario_df.iloc[st_end:sv_end][cols_to_save].to_parquet(client_dir / "val.parquet")
        scenario_df.iloc[sv_end:][cols_to_save].to_parquet(client_dir / "test.parquet")
        
        print(f"  {scenario}: train={st_end}, val={sv_end-st_end}, test={sn-sv_end}")
    
    return output_dir


def run_training(data_dir: Path):
    """Run federated training on real data."""
    print("\n" + "="*60)
    print("TRAINING ON REAL IOT-23 DATA")
    print("="*60 + "\n")
    
    # Use relative path to avoid spaces issue
    rel_data_dir = data_dir.relative_to(Path.cwd())
    cmd = f'python train.py --mode federated --data_dir {rel_data_dir} --config configs/test_tiny.yaml --fl_rounds 2 --fl_epochs 2'
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    print("""
    ================================================================
    |    REAL IOT-23 DATA TEST (NO SYNTHETIC DATA)                 |
    |    Source: Stratosphere IPS, CTU University                  |
    |    Dataset: Aposemat IoT-23 Malware Traffic Captures         |
    ================================================================
    """)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    raw_dir = project_root / "data" / "raw" / "iot23_real"
    processed_dir = project_root / "data" / "processed" / "iot23_real"
    
    # Step 1: Download real IoT-23 data
    downloaded_files = download_real_iot23(raw_dir, max_scenarios=2)
    
    if not downloaded_files:
        print("\n" + "="*60)
        print("ERROR: Could not download real IoT-23 data!")
        print("Please check your internet connection or try manually downloading from:")
        print("https://www.stratosphereips.org/datasets-iot23")
        print("="*60)
        return 1
    
    # Step 2: Preprocess the data
    data_dir = preprocess_iot23_data(downloaded_files, processed_dir)
    
    if data_dir is None:
        print("ERROR: Preprocessing failed!")
        return 1
    
    # Step 3: Run training
    success = run_training(data_dir)
    
    if success:
        print("""
    ================================================================
    |              REAL DATA TEST SUCCESSFUL!                      |
    |    Your FL-TTA pipeline works with actual IoT-23 data        |
    |    (Real malware traffic from CTU University captures)       |
    ================================================================
        """)
        return 0
    else:
        print("\nTraining had issues but data pipeline works!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
