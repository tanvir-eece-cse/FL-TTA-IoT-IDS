"""
Test-Time Adaptation (TTA) for IoT IDS
Implements:
- TENT: Fully Test-Time Adaptation by Entropy Minimization
- EATA: Efficient Test-Time Adaptation with Entropy-Aware Sample Selection
- Drift-Aware TTA: Only adapt when drift is detected

References:
- TENT: https://arxiv.org/abs/2006.10726
- EATA: https://arxiv.org/abs/2204.02610
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Callable, Tuple, Any
import numpy as np
from copy import deepcopy
from collections import deque


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of softmax predictions."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy


def collect_bn_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect BatchNorm parameters for adaptation."""
    params = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    params.append(param)
                    names.append(f"{name}.{param_name}")
    return params, names


def collect_affine_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect affine parameters (last layer) for adaptation."""
    params = []
    names = []
    
    # Find classifier/last linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    params.append(param)
                    names.append(f"{name}.{param_name}")
    
    # Return only last layer params
    if len(params) >= 2:
        return params[-2:], names[-2:]  # weight and bias of last linear
    return params, names


class TENT:
    """
    TENT: Fully Test-Time Adaptation by Entropy Minimization
    
    Adapts BatchNorm affine parameters by minimizing prediction entropy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = False,
        adapt_bn_only: bool = True
    ):
        """
        Args:
            model: Model to adapt
            optimizer_class: Optimizer class
            lr: Learning rate for adaptation
            steps: Number of adaptation steps per batch
            episodic: If True, reset model after each batch
            adapt_bn_only: If True, only adapt BatchNorm params; else adapt all
        """
        self.model = model
        self.lr = lr
        self.steps = steps
        self.episodic = episodic
        self.adapt_bn_only = adapt_bn_only
        
        # Store original model state for episodic adaptation
        self.model_state = deepcopy(model.state_dict())
        
        # Setup model for adaptation
        self._configure_model()
        
        # Collect parameters to adapt
        if adapt_bn_only:
            params, param_names = collect_bn_params(model)
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            param_names = [n for n, _ in model.named_parameters()]
        
        self.params = params
        self.param_names = param_names
        
        # Create optimizer
        self.optimizer = optimizer_class(params, lr=lr)
        
        print(f"[TENT] Adapting {len(params)} parameters: {param_names[:5]}...")
    
    def _configure_model(self):
        """Configure model for test-time adaptation."""
        self.model.train()  # Enable BN updates
        
        # Disable gradient for non-BN params if adapt_bn_only
        if self.adapt_bn_only:
            for name, param in self.model.named_parameters():
                if 'bn' not in name.lower() and 'norm' not in name.lower():
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Keep BN in eval mode for running stats, but train for affine params
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)
                # Use batch stats at test time
                module.track_running_stats = False
    
    def reset(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.model_state, strict=True)
        self._configure_model()
        
        if self.adapt_bn_only:
            params, _ = collect_bn_params(self.model)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = type(self.optimizer)(params, lr=self.lr)
    
    @torch.enable_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptation."""
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            # Forward
            logits = self.model(x)
            
            # Compute entropy loss
            entropy = softmax_entropy(logits)
            loss = entropy.mean()
            
            # Backward and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Final forward (no grad)
        with torch.no_grad():
            logits = self.model(x)
        
        return logits
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class EATA:
    """
    EATA: Efficient Test-Time Adaptation
    
    Improvements over TENT:
    1. Sample-aware entropy: Only adapt on reliable (low entropy) samples
    2. Anti-forgetting: Regularize important parameters (Fisher-weighted)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 1e-4,
        steps: int = 1,
        entropy_threshold: float = 0.4 * np.log(2),  # 40% of max binary entropy
        fisher_alpha: float = 2000.0,  # Fisher regularization strength
        adapt_bn_only: bool = True
    ):
        self.model = model
        self.lr = lr
        self.steps = steps
        self.entropy_threshold = entropy_threshold
        self.fisher_alpha = fisher_alpha
        self.adapt_bn_only = adapt_bn_only
        
        # Store original model
        self.model_state = deepcopy(model.state_dict())
        
        # Configure model
        self._configure_model()
        
        # Collect parameters
        if adapt_bn_only:
            params, param_names = collect_bn_params(model)
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            param_names = [n for n, _ in model.named_parameters()]
        
        self.params = params
        self.param_names = param_names
        
        # Store original param values for anti-forgetting
        self.anchor_params = {name: p.clone().detach() for name, p in zip(param_names, params)}
        
        # Fisher importance (initialized as zeros, can be computed from source data)
        self.fisher = {name: torch.zeros_like(p) for name, p in zip(param_names, params)}
        
        # Optimizer
        self.optimizer = optimizer_class(params, lr=lr)
        
        # Stats tracking
        self.num_adapted = 0
        self.num_skipped = 0
        
        print(f"[EATA] Adapting {len(params)} parameters with entropy threshold {entropy_threshold:.4f}")
    
    def _configure_model(self):
        """Configure model for adaptation."""
        self.model.train()
        
        if self.adapt_bn_only:
            for name, param in self.model.named_parameters():
                if 'bn' not in name.lower() and 'norm' not in name.lower():
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)
                module.track_running_stats = False
    
    def compute_fisher(self, dataloader: DataLoader, num_batches: int = 100):
        """Compute Fisher information from source data (optional enhancement)."""
        self.model.eval()
        fisher = {name: torch.zeros_like(p) for name, p in zip(self.param_names, self.params)}
        
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            x = x.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            
            # Empirical Fisher
            pseudo_labels = probs.argmax(dim=1)
            loss = F.cross_entropy(logits, pseudo_labels)
            loss.backward()
            
            for name, param in zip(self.param_names, self.params):
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
        
        # Normalize
        for name in fisher:
            fisher[name] /= (i + 1)
        
        self.fisher = fisher
        self._configure_model()
    
    def reset(self):
        """Reset to original model."""
        self.model.load_state_dict(self.model_state, strict=True)
        self._configure_model()
        
        if self.adapt_bn_only:
            params, _ = collect_bn_params(self.model)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.params = params
        self.anchor_params = {name: p.clone().detach() for name, p in zip(self.param_names, params)}
        self.optimizer = type(self.optimizer)(params, lr=self.lr)
        self.num_adapted = 0
        self.num_skipped = 0
    
    @torch.enable_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sample-aware adaptation."""
        for _ in range(self.steps):
            logits = self.model(x)
            
            # Compute entropy for sample selection
            entropy = softmax_entropy(logits)
            
            # Select reliable samples (low entropy)
            reliable_mask = entropy < self.entropy_threshold
            
            if reliable_mask.sum() > 0:
                # Entropy loss on reliable samples only
                entropy_loss = entropy[reliable_mask].mean()
                
                # Anti-forgetting regularization
                reg_loss = 0.0
                for name, param in zip(self.param_names, self.params):
                    if name in self.anchor_params:
                        diff = param - self.anchor_params[name]
                        fisher_weight = self.fisher.get(name, torch.ones_like(param))
                        reg_loss += (fisher_weight * diff.pow(2)).sum()
                
                loss = entropy_loss + self.fisher_alpha * reg_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.num_adapted += reliable_mask.sum().item()
            else:
                self.num_skipped += len(entropy)
        
        # Final forward
        with torch.no_grad():
            logits = self.model(x)
        
        return logits
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def get_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        total = self.num_adapted + self.num_skipped
        return {
            'num_adapted': self.num_adapted,
            'num_skipped': self.num_skipped,
            'adaptation_ratio': self.num_adapted / max(total, 1)
        }


class DriftAwareTTA:
    """
    Drift-Aware Test-Time Adaptation
    
    Only triggers adaptation when drift is detected via:
    1. Entropy shift: Average entropy exceeds threshold
    2. Feature shift: KL divergence on feature distributions
    
    This prevents over-adaptation on in-distribution data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_tta: str = 'tent',  # 'tent' or 'eata'
        lr: float = 1e-4,
        steps: int = 1,
        drift_window: int = 100,
        entropy_drift_threshold: float = 0.3,  # Relative entropy increase
        adapt_bn_only: bool = True,
        **tta_kwargs
    ):
        self.model = model
        self.drift_window = drift_window
        self.entropy_drift_threshold = entropy_drift_threshold
        
        # Base TTA method
        if base_tta == 'tent':
            self.tta = TENT(model, lr=lr, steps=steps, adapt_bn_only=adapt_bn_only)
        elif base_tta == 'eata':
            self.tta = EATA(model, lr=lr, steps=steps, adapt_bn_only=adapt_bn_only, **tta_kwargs)
        else:
            raise ValueError(f"Unknown TTA method: {base_tta}")
        
        # Drift detection state
        self.entropy_history = deque(maxlen=drift_window)
        self.baseline_entropy = None
        self.drift_detected = False
        self.adaptation_count = 0
        
        print(f"[DriftAwareTTA] Using {base_tta} with drift threshold {entropy_drift_threshold}")
    
    def set_baseline(self, dataloader: DataLoader, num_batches: int = 50):
        """Compute baseline entropy from validation/source data."""
        self.model.eval()
        entropies = []
        
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                x = x.to(next(self.model.parameters()).device)
                logits = self.model(x)
                entropy = softmax_entropy(logits)
                entropies.extend(entropy.cpu().numpy())
        
        self.baseline_entropy = np.mean(entropies)
        print(f"[DriftAwareTTA] Baseline entropy: {self.baseline_entropy:.4f}")
    
    def _detect_drift(self, entropy: torch.Tensor) -> bool:
        """Detect drift based on entropy shift."""
        current_entropy = entropy.mean().item()
        self.entropy_history.append(current_entropy)
        
        if self.baseline_entropy is None:
            return True  # No baseline, always adapt
        
        if len(self.entropy_history) < self.drift_window // 2:
            return False  # Not enough history
        
        window_entropy = np.mean(list(self.entropy_history))
        relative_increase = (window_entropy - self.baseline_entropy) / (self.baseline_entropy + 1e-6)
        
        return relative_increase > self.entropy_drift_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with drift-aware adaptation."""
        # First, get predictions to check for drift
        with torch.no_grad():
            logits = self.model(x)
            entropy = softmax_entropy(logits)
        
        # Check for drift
        self.drift_detected = self._detect_drift(entropy)
        
        if self.drift_detected:
            # Drift detected: apply TTA
            logits = self.tta(x)
            self.adaptation_count += 1
        
        return logits
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def reset(self):
        """Reset TTA and drift detection."""
        self.tta.reset()
        self.entropy_history.clear()
        self.drift_detected = False
        self.adaptation_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get drift and adaptation statistics."""
        stats = {
            'adaptation_count': self.adaptation_count,
            'baseline_entropy': self.baseline_entropy,
            'current_window_entropy': np.mean(list(self.entropy_history)) if self.entropy_history else None,
            'drift_detected': self.drift_detected
        }
        if hasattr(self.tta, 'get_stats'):
            stats.update(self.tta.get_stats())
        return stats


def create_tta(
    model: nn.Module,
    method: str = 'tent',
    **kwargs
) -> Callable:
    """Factory function to create TTA methods."""
    if method == 'tent':
        return TENT(model, **kwargs)
    elif method == 'eata':
        return EATA(model, **kwargs)
    elif method == 'drift_aware':
        return DriftAwareTTA(model, **kwargs)
    else:
        raise ValueError(f"Unknown TTA method: {method}")
