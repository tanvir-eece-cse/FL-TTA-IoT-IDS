"""
Lightning AI L40S Setup Script
================================
Automatically sets up the environment on Lightning AI Studios

Run this first on Lightning AI:
    python setup_lightning.py
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    print(f"\n✅ Success: {description}")
    return True

def main():
    print("""
    ================================================================
    |  FL-TTA-IoT-IDS: Lightning AI L40S Setup                     |
    |  Automated Environment Configuration                         |
    ================================================================
    """)
    
    # Check if on Lightning AI
    is_lightning = os.environ.get('LIGHTNING_CLOUD_URL') or os.environ.get('LIGHTNING_STUDIO')
    if is_lightning:
        print("✓ Running on Lightning AI")
    else:
        print("⚠ Not detected as Lightning AI (continuing anyway)")
    
    # Install requirements
    if not run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    ):
        return 1
    
    # Check GPU
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ VRAM: {vram:.1f} GB")
        
        if 'L40S' in gpu_name:
            print("✓ L40S GPU confirmed - optimal configuration")
        elif 'A100' in gpu_name:
            print("✓ A100 GPU detected - excellent performance expected")
        else:
            print(f"⚠ GPU is {gpu_name}, not L40S - may need config adjustments")
    else:
        print("❌ No GPU detected!")
        return 1
    
    # Download small test data if not exists
    data_dir = Path("data/processed/iot23_real")
    if not data_dir.exists():
        print("\n" + "="*60)
        print("DOWNLOADING TEST DATA")
        print("="*60)
        
        if not run_command(
            "python test_real_iot23.py",
            "Downloading real IoT-23 test data (2 scenarios)"
        ):
            print("\n⚠ Data download failed, but setup otherwise complete")
            print("You can manually download data later with:")
            print("  python test_real_iot23.py")
    else:
        print(f"\n✓ Data already exists at: {data_dir}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Get your WandB API key from: https://wandb.ai/authorize")
    print("2. Run quick training:")
    print("   python train_lightning_quick.py --wandb-key YOUR_KEY")
    print("\nOr run without WandB:")
    print("   python train_lightning_quick.py --skip-wandb")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
