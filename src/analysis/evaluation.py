"""
Evaluation and Analysis Pipeline
Generates metrics, plots, and tables for the research paper
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from collections import defaultdict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Binary-specific metrics
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        if y_prob is not None:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def evaluate_model_on_loader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    tta_method: Optional[callable] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            if tta_method is not None:
                logits = tta_method(x)
            else:
                logits = model(x)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            
            # For binary, use prob of positive class
            if probs.shape[1] == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
):
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict,
    save_path: Optional[Path] = None
):
    """Plot training loss and metrics curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.get('round', range(len(history.get('train_loss', [])))), 
                 history.get('train_loss', []), label='Train')
    axes[0].plot(history.get('round', range(len(history.get('val_loss', [])))),
                 history.get('val_loss', []), label='Val')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'val_accuracy' in history:
        axes[1].plot(history.get('round', range(len(history['val_accuracy']))),
                     history['val_accuracy'])
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1_macro' in history:
        axes[2].plot(history.get('round', range(len(history['val_f1_macro']))),
                     history['val_f1_macro'])
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('F1 Macro')
        axes[2].set_title('Validation F1 Score')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_client_comparison(
    client_metrics: Dict[str, Dict[str, float]],
    metric: str = 'f1_macro',
    save_path: Optional[Path] = None,
    title: str = "Per-Client Performance"
):
    """Plot bar chart comparing client performance."""
    clients = list(client_metrics.keys())
    values = [client_metrics[c].get(metric, 0) for c in clients]
    
    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(len(clients)), values, color='steelblue')
    plt.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
    plt.xticks(range(len(clients)), clients, rotation=45, ha='right')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_drift_analysis(
    window_metrics: List[Dict[str, float]],
    window_size: int,
    save_path: Optional[Path] = None
):
    """Plot performance over time windows to visualize drift."""
    windows = list(range(len(window_metrics)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # F1 over time
    f1_values = [m.get('f1_macro', 0) for m in window_metrics]
    axes[0].plot(windows, f1_values, 'b-o', markersize=4)
    axes[0].set_xlabel(f'Time Window (size={window_size})')
    axes[0].set_ylabel('F1 Macro')
    axes[0].set_title('F1 Score Over Time (Drift Analysis)')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy over time
    acc_values = [m.get('accuracy', 0) for m in window_metrics]
    axes[1].plot(windows, acc_values, 'g-o', markersize=4)
    axes[1].set_xlabel(f'Time Window (size={window_size})')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Time (Drift Analysis)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'auroc'],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """Generate comparison table for different methods."""
    df = pd.DataFrame(results).T
    df = df[metrics]
    df = df.round(4)
    
    # Highlight best values
    styled = df.style.highlight_max(axis=0, color='lightgreen')
    
    if save_path:
        df.to_csv(save_path)
        # Also save LaTeX version for paper
        latex_path = save_path.with_suffix('.tex')
        df.to_latex(latex_path, caption="Performance Comparison", label="tab:comparison")
    
    return df


def generate_paper_results(
    experiment_dir: Path,
    output_dir: Optional[Path] = None
):
    """Generate all figures and tables for the paper."""
    if output_dir is None:
        output_dir = experiment_dir / "paper_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    history_path = experiment_dir / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Training curves
        plot_training_curves(history, output_dir / "training_curves.png")
    
    # Load and plot client metrics if available
    if 'tta_results' in history:
        plot_client_comparison(
            history['tta_results'],
            metric='f1_macro',
            save_path=output_dir / "client_comparison_f1.png",
            title="Per-Client F1 Score with TTA"
        )
        plot_client_comparison(
            history['tta_results'],
            metric='accuracy',
            save_path=output_dir / "client_comparison_acc.png",
            title="Per-Client Accuracy with TTA"
        )
    
    print(f"Paper outputs saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    generate_paper_results(
        Path(args.experiment_dir),
        Path(args.output_dir) if args.output_dir else None
    )
