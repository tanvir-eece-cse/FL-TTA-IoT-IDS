"""
IoT-23 Dataset Download Script
Downloads the small labeled Zeek flows from Stratosphere IPS
Dataset: https://www.stratosphereips.org/datasets-iot23
"""
import os
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import hashlib

# IoT-23 Base URL and scenario configurations
# The small tar.gz (8.7 GB) contains all labeled conn.log files
# Individual scenarios can be downloaded separately

IOT23_BASE_URL = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset"

# Full dataset download (recommended for production)
IOT23_FULL_URL = f"{IOT23_BASE_URL}/iot_23_datasets_small.tar.gz"

# Individual scenario URLs (from the webpage listing)
IOT23_URLS = {
    # Small subset - 4 diverse scenarios for quick experiments
    "small": {
        "CTU-IoT-Malware-Capture-34-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/",
        "CTU-IoT-Malware-Capture-8-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-8-1/",
        "CTU-Honeypot-Capture-4-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-Honeypot-Capture-4-1/",
        "CTU-Honeypot-Capture-5-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-Honeypot-Capture-5-1/",
    },
    # Medium - more scenarios for better FL simulation
    "medium": {
        "CTU-IoT-Malware-Capture-34-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/",
        "CTU-IoT-Malware-Capture-43-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-43-1/",
        "CTU-IoT-Malware-Capture-7-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-7-1/",
        "CTU-IoT-Malware-Capture-8-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-8-1/",
        "CTU-IoT-Malware-Capture-9-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-9-1/",
        "CTU-IoT-Malware-Capture-17-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-17-1/",
        "CTU-IoT-Malware-Capture-33-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-33-1/",
        "CTU-IoT-Malware-Capture-35-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-IoT-Malware-Capture-35-1/",
        "CTU-Honeypot-Capture-4-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-Honeypot-Capture-4-1/",
        "CTU-Honeypot-Capture-5-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-Honeypot-Capture-5-1/",
        "CTU-Honeypot-Capture-7-1": f"{IOT23_BASE_URL}/IndividualScenarios/CTU-Honeypot-Capture-7-1/",
    }
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def decompress_gz(gz_path: Path, output_path: Path) -> bool:
    """Decompress .gz file."""
    import gzip
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return True
    except Exception as e:
        print(f"Error decompressing {gz_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download IoT-23 Dataset")
    parser.add_argument("--subset", type=str, default="small", choices=["small", "medium", "full"],
                        help="Dataset subset size (small=4 scenarios, medium=11 scenarios, full=all)")
    parser.add_argument("--output_dir", type=str, default="data/raw/iot23",
                        help="Output directory for downloaded data")
    parser.add_argument("--keep_compressed", action="store_true",
                        help="Keep compressed files after extraction")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.subset == "full":
        # Download the full small tar.gz (8.7 GB labeled flows only)
        print("Downloading full IoT-23 dataset (8.7 GB)...")
        tar_path = output_dir / "iot_23_datasets_small.tar.gz"
        
        if not tar_path.exists():
            success = download_file(IOT23_FULL_URL, tar_path)
            if not success:
                print("Failed to download full dataset")
                return
        
        # Extract
        print("Extracting...")
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        print(f"Extracted to: {output_dir}")
        return
    
    # Download individual scenarios
    scenarios = IOT23_URLS[args.subset]
    print(f"Downloading IoT-23 {args.subset} subset ({len(scenarios)} scenarios)...")
    
    for scenario_name, base_url in scenarios.items():
        print(f"\n[{scenario_name}]")
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = scenario_dir / "conn.log.labeled"
        
        # Skip if already exists
        if log_path.exists():
            print(f"  Already exists: {log_path}")
            continue
        
        # Try direct conn.log.labeled file first
        direct_url = f"{base_url}conn.log.labeled"
        print(f"  Trying: {direct_url}")
        
        success = download_file(direct_url, log_path)
        
        if not success:
            # Try bro/conn.log.labeled path
            bro_url = f"{base_url}bro/conn.log.labeled"
            print(f"  Trying: {bro_url}")
            success = download_file(bro_url, log_path)
        
        if success:
            file_size = log_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded: {file_size:.1f} MB")
        else:
            print(f"  ✗ Failed to download {scenario_name}")
        
    print(f"\nDownload complete! Data saved to: {output_dir}")
    print(f"Next step: Run preprocessing with:")
    print(f"  python scripts/preprocess.py --input_dir {output_dir}")
    print(f"  python scripts/preprocess.py --input_dir {output_dir}")


if __name__ == "__main__":
    main()
