"""
Federated Learning Implementation using Flower
Supports:
- FedAvg (Federated Averaging)
- FedProx (with proximal term)
- Non-IID client data partitioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, OrderedDict
from pathlib import Path
import numpy as np
from collections import OrderedDict as OD
import copy

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, 
    ndarrays_to_parameters, parameters_to_ndarrays
)

from src.models import create_model
from src.data import IoT23Dataset


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]):
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OD({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class IoTFlowerClient(fl.client.NumPyClient):
    """
    Flower client for IoT IDS federated learning.
    Represents a single IoT device/scenario.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        learning_rate: float = 1e-3,
        proximal_mu: float = 0.0  # FedProx regularization (0 = FedAvg)
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.proximal_mu = proximal_mu
        
        self.model.to(device)
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return get_parameters(self.model)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train model on local data."""
        # Set global model parameters
        set_parameters(self.model, parameters)
        
        # Store global params for FedProx
        if self.proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Local training
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                
                # FedProx proximal term
                if self.proximal_mu > 0:
                    proximal_loss = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params):
                        proximal_loss += ((param - global_param) ** 2).sum()
                    loss += (self.proximal_mu / 2) * proximal_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / total_samples
        
        return (
            get_parameters(self.model),
            total_samples,
            {"loss": float(avg_loss), "client_id": self.client_id}
        )
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data."""
        set_parameters(self.model, parameters)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                
                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Compute F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return (
            float(avg_loss),
            total,
            {
                "accuracy": float(accuracy),
                "f1_macro": float(f1),
                "client_id": self.client_id
            }
        )


class FederatedAggregator:
    """
    Server-side aggregation logic for federated learning.
    Supports weighted averaging based on client data sizes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_rounds: int = 50,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2
    ):
        self.model = model
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # History tracking
        self.history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'f1_macro': []
        }
    
    def aggregate_fit(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict[str, Scalar]]]
    ) -> List[np.ndarray]:
        """Aggregate model updates using weighted averaging."""
        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples, _ in results)
        
        # Weighted average
        aggregated = None
        for params, num_samples, metrics in results:
            weight = num_samples / total_samples
            if aggregated is None:
                aggregated = [weight * np.array(p) for p in params]
            else:
                for i, p in enumerate(params):
                    aggregated[i] += weight * np.array(p)
        
        return aggregated
    
    def aggregate_evaluate(
        self,
        results: List[Tuple[float, int, Dict[str, Scalar]]]
    ) -> Dict[str, float]:
        """Aggregate evaluation metrics."""
        total_samples = sum(num_samples for _, num_samples, _ in results)
        
        weighted_loss = sum(loss * n for loss, n, _ in results) / total_samples
        weighted_acc = sum(m.get('accuracy', 0) * n for _, n, m in results) / total_samples
        weighted_f1 = sum(m.get('f1_macro', 0) * n for _, n, m in results) / total_samples
        
        return {
            'loss': weighted_loss,
            'accuracy': weighted_acc,
            'f1_macro': weighted_f1
        }


def run_federated_simulation(
    model: nn.Module,
    client_data: Dict[str, Tuple[DataLoader, DataLoader]],  # {client_id: (train_loader, val_loader)}
    num_rounds: int = 50,
    local_epochs: int = 5,
    learning_rate: float = 1e-3,
    proximal_mu: float = 0.0,
    device: torch.device = torch.device('cuda'),
    save_dir: Optional[Path] = None
) -> Tuple[nn.Module, Dict]:
    """
    Run federated learning simulation.
    
    Args:
        model: Global model to train
        client_data: Dictionary mapping client IDs to (train_loader, val_loader)
        num_rounds: Number of FL rounds
        local_epochs: Epochs per client per round
        learning_rate: Client learning rate
        proximal_mu: FedProx regularization (0 = FedAvg)
        device: Training device
        save_dir: Directory to save checkpoints
    
    Returns:
        Trained model and history dict
    """
    # Create clients
    clients = {}
    for client_id, (train_loader, val_loader) in client_data.items():
        client_model = copy.deepcopy(model)
        clients[client_id] = IoTFlowerClient(
            client_id=client_id,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            proximal_mu=proximal_mu
        )
    
    # Initialize global parameters
    global_params = get_parameters(model)
    
    # Aggregator
    aggregator = FederatedAggregator(model, num_rounds)
    
    # Training loop
    history = {
        'round': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': [],
        'client_metrics': []
    }
    
    best_f1 = 0.0
    best_params = None
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning: {num_rounds} rounds, {len(clients)} clients")
    print(f"{'='*60}\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")
        
        # Client training
        fit_results = []
        for client_id, client in clients.items():
            params, num_samples, metrics = client.fit(global_params, {})
            fit_results.append((params, num_samples, metrics))
        
        # Aggregate
        global_params = aggregator.aggregate_fit(fit_results)
        
        # Update global model
        set_parameters(model, global_params)
        
        # Evaluate
        eval_results = []
        for client_id, client in clients.items():
            loss, num_samples, metrics = client.evaluate(global_params, {})
            eval_results.append((loss, num_samples, metrics))
        
        # Aggregate metrics
        metrics = aggregator.aggregate_evaluate(eval_results)
        
        # Log
        train_loss = np.mean([r[2]['loss'] for r in fit_results])
        history['round'].append(round_num)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_f1_macro'].append(metrics['f1_macro'])
        history['client_metrics'].append([r[2] for r in eval_results])
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
        
        # Save best model
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            best_params = copy.deepcopy(global_params)
            print(f"  ★ New best F1: {best_f1:.4f}")
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "best_model.pt")
    
    # Load best parameters
    if best_params is not None:
        set_parameters(model, best_params)
    
    print(f"\n{'='*60}")
    print(f"Federated Learning Complete!")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"{'='*60}\n")
    
    return model, history
