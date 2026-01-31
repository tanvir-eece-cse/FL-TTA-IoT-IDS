"""
Lightning Module for IoT IDS Training
Supports centralized and federated training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ConfusionMatrix
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from typing import Optional, Dict, Any, List
import numpy as np

from .networks import create_model


class IoTIDSLightningModule(L.LightningModule):
    """
    Lightning Module for IoT Intrusion Detection.
    Optimized for L40S GPU with:
    - Mixed precision support
    - Gradient accumulation
    - Learning rate scheduling
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        architecture: str = 'mlp',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Create model
        self.model = create_model(
            input_dim=input_dim,
            num_classes=num_classes,
            architecture=architecture,
            **model_kwargs
        )
        
        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weight', weight)
        else:
            self.class_weight = None
        
        self.label_smoothing = label_smoothing
        
        # Metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup torchmetrics for train/val/test."""
        task = 'binary' if self.num_classes == 2 else 'multiclass'
        
        for stage in ['train', 'val', 'test']:
            if task == 'binary':
                setattr(self, f'{stage}_acc', Accuracy(task='binary'))
                setattr(self, f'{stage}_f1', F1Score(task='binary'))
                setattr(self, f'{stage}_precision', Precision(task='binary'))
                setattr(self, f'{stage}_recall', Recall(task='binary'))
                setattr(self, f'{stage}_auroc', AUROC(task='binary'))
            else:
                setattr(self, f'{stage}_acc', Accuracy(task='multiclass', num_classes=self.num_classes))
                setattr(self, f'{stage}_f1', F1Score(task='multiclass', num_classes=self.num_classes, average='macro'))
                setattr(self, f'{stage}_precision', Precision(task='multiclass', num_classes=self.num_classes, average='macro'))
                setattr(self, f'{stage}_recall', Recall(task='multiclass', num_classes=self.num_classes, average='macro'))
                setattr(self, f'{stage}_auroc', AUROC(task='multiclass', num_classes=self.num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with optional class weights and label smoothing."""
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing
        )
    
    def _shared_step(self, batch, stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        
        # Get predictions
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        acc_metric = getattr(self, f'{stage}_acc')
        f1_metric = getattr(self, f'{stage}_f1')
        precision_metric = getattr(self, f'{stage}_precision')
        recall_metric = getattr(self, f'{stage}_recall')
        auroc_metric = getattr(self, f'{stage}_auroc')
        
        acc_metric(preds, y)
        f1_metric(preds, y)
        precision_metric(preds, y)
        recall_metric(preds, y)
        
        if self.num_classes == 2:
            auroc_metric(probs, y)
        else:
            auroc_metric(probs, y)
        
        return {'loss': loss, 'logits': logits, 'preds': preds, 'probs': probs}
    
    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, 'train')
        
        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, 'val')
        
        self.log('val/loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True)
        
        return outputs['loss']
    
    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, 'test')
        
        self.log('test/loss', outputs['loss'], on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True)
        
        return outputs
    
    def configure_optimizers(self):
        """Configure optimizer with cosine annealing + warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def get_model(self) -> nn.Module:
        """Return the underlying model for FL or TTA."""
        return self.model
