from .evaluation import (
    compute_metrics, evaluate_model_on_loader,
    plot_confusion_matrix, plot_roc_curve, plot_training_curves,
    plot_client_comparison, plot_drift_analysis,
    generate_comparison_table, generate_paper_results
)

__all__ = [
    'compute_metrics', 'evaluate_model_on_loader',
    'plot_confusion_matrix', 'plot_roc_curve', 'plot_training_curves',
    'plot_client_comparison', 'plot_drift_analysis',
    'generate_comparison_table', 'generate_paper_results'
]
