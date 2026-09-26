"""
abCAN-v2 — Metrics
Pearson r, Spearman ρ, RMSE, and Bootstrap CIs.
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, roc_auc_score
from typing import Dict, Tuple

def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate main regression and classification metrics."""
    # Regression
    pearson_r, _ = stats.pearsonr(preds, targets)
    spearman_rho, _ = stats.spearmanr(preds, targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    
    # Classification (Stabilizing vs Destabilizing, ΔΔG < 0 vs > 0)
    # Target < 0 is usually stabilizing (positive class)
    binary_targets = (targets < 0).astype(int)
    # Using negative pred as score (lower pred = more stabilizing = higher score)
    try:
        auroc = roc_auc_score(binary_targets, -preds)
    except ValueError:
        auroc = float('nan') # Handle edge cases with only one class
        
    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
        "auroc": auroc
    }

def bootstrap_ci(
    preds: np.ndarray, 
    targets: np.ndarray, 
    n_iterations: int = 1000, 
    ci: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """Calculate Bootstrap Confidence Intervals for metrics."""
    n_size = len(preds)
    metrics_dist = {"pearson_r": [], "spearman_rho": [], "rmse": []}
    
    for i in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, n_size, n_size)
        sample_preds = preds[indices]
        sample_targets = targets[indices]
        
        # Skip if all targets are the same (happens rarely in small samples)
        if len(np.unique(sample_targets)) < 2:
            continue
            
        m = calculate_metrics(sample_preds, sample_targets)
        for k in metrics_dist.keys():
            metrics_dist[k].append(m[k])
            
    # Calculate percentiles
    lower = (1.0 - ci) / 2.0 * 100
    upper = (1.0 + ci) / 2.0 * 100
    
    results = {}
    for k, v in metrics_dist.items():
        results[k] = (np.percentile(v, lower), np.percentile(v, upper))
        
    return results
