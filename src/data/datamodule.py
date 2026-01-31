"""
IoT-23 DataModule for PyTorch Lightning
Handles data loading with optimizations for L40S GPU
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import lightning as L
from sklearn.preprocessing import StandardScaler
import joblib


class IoT23Dataset(Dataset):
    """IoT-23 Dataset for tabular classification."""
    
    def __init__(
        self,
        data_path: Path,
        feature_columns: List[str],
        label_column: str = 'label_binary',
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        self.data_path = Path(data_path)
        self.feature_columns = feature_columns
        self.label_column = label_column
        
        # Load data
        self.df = pd.read_parquet(self.data_path)
        
        # Extract features and labels
        self.features = self.df[feature_columns].values.astype(np.float32)
        self.labels = self.df[label_column].values
        
        # Handle multiclass labels
        if self.labels.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.labels)
        
        self.labels = self.labels.astype(np.int64)
        
        # Normalize features
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Replace NaN/Inf with 0
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    @property
    def num_features(self):
        return self.features.shape[1]
    
    @property
    def num_classes(self):
        return len(torch.unique(self.labels))


class IoT23DataModule(L.LightningDataModule):
    """Lightning DataModule for IoT-23 with L40S optimizations."""
    
    def __init__(
        self,
        data_dir: str = "data/processed/iot23",
        batch_size: int = 4096,  # Large batch for L40S
        num_workers: int = 8,
        label_column: str = 'label_binary',
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_column = label_column
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        # Load feature columns
        with open(self.data_dir / "feature_columns.txt", 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        self.scaler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Train dataset (fit scaler)
            self.train_dataset = IoT23Dataset(
                self.data_dir / "train.parquet",
                self.feature_columns,
                self.label_column,
                fit_scaler=True
            )
            self.scaler = self.train_dataset.scaler
            
            # Validation dataset (use train scaler)
            self.val_dataset = IoT23Dataset(
                self.data_dir / "val.parquet",
                self.feature_columns,
                self.label_column,
                scaler=self.scaler
            )
        
        if stage == "test" or stage is None:
            # Test dataset
            if self.scaler is None:
                # Load scaler from train if not already loaded
                self.train_dataset = IoT23Dataset(
                    self.data_dir / "train.parquet",
                    self.feature_columns,
                    self.label_column,
                    fit_scaler=True
                )
                self.scaler = self.train_dataset.scaler
            
            self.test_dataset = IoT23Dataset(
                self.data_dir / "test.parquet",
                self.feature_columns,
                self.label_column,
                scaler=self.scaler
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,  # Larger batch for eval
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    @property
    def num_features(self):
        return len(self.feature_columns)
    
    @property
    def num_classes(self):
        if self.train_dataset:
            return self.train_dataset.num_classes
        return 2  # Default binary


class FederatedClientDataModule(L.LightningDataModule):
    """DataModule for a single federated learning client."""
    
    def __init__(
        self,
        client_dir: str,
        feature_columns: List[str],
        batch_size: int = 2048,
        num_workers: int = 4,
        label_column: str = 'label_binary',
        scaler: Optional[StandardScaler] = None
    ):
        super().__init__()
        self.client_dir = Path(client_dir)
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_column = label_column
        self.scaler = scaler
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = IoT23Dataset(
                self.client_dir / "train.parquet",
                self.feature_columns,
                self.label_column,
                scaler=self.scaler,
                fit_scaler=(self.scaler is None)
            )
            if self.scaler is None:
                self.scaler = self.train_dataset.scaler
            
            self.val_dataset = IoT23Dataset(
                self.client_dir / "val.parquet",
                self.feature_columns,
                self.label_column,
                scaler=self.scaler
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = IoT23Dataset(
                self.client_dir / "test.parquet",
                self.feature_columns,
                self.label_column,
                scaler=self.scaler
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
