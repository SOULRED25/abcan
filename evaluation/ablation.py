"""
abCAN-v2 — Ablation Study Runner
Temporarily disables structural, sequence, or biochemical branches.
"""

import copy
import logging
import torch
import torch.nn as nn
from typing import Dict, Any

logger = logging.getLogger(__name__)


def disable_module(model: nn.Module, component: str):
    """Zeroes out the output of a specified architectural component."""
    
    class ZeroOutModule(nn.Module):
        def forward(self, *args, **kwargs):
            # Return zeros in the shape of the first tensor argument
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    return torch.zeros_like(arg)
            return None

    if component == "mpnn_structural":
        model.structural_encoder = ZeroOutModule()
    elif component == "mpnn_sequence":
        model.sequence_encoder = ZeroOutModule()
    elif component == "physicochemical":
        # Modify the injection module to just return the main_features unmodified
        class IdentityInjection(nn.Module):
            def forward(self, main_features, biochem_features):
                return main_features
        model.physicochemical_injection = IdentityInjection()
    elif component == "aggregation":
        # Force non-attention weighted aggregation (mean pooling fallback)
        model.aggregation.attention_weighted = False
    else:
        raise ValueError(f"Unknown component for ablation: {component}")
        

def run_ablation_experiment(
    base_model: nn.Module, 
    component_to_remove: str,
    test_loader, 
    device: str
) -> Dict[str, float]:
    """
    Run an ablation test by disabling a component.
    (Note: In a true experimental setup, you'd retrain the model.
    This simulates zeroing it out during inference/evaluation).
    """
    logger.info(f"Running Ablation: Removing {component_to_remove}")
    
    # Deepcopy to avoid modifying the original model
    ablated_model = copy.deepcopy(base_model)
    disable_module(ablated_model, component_to_remove)
    
    ablated_model.eval()
    
    # ... logic to run test_loader and collect predictions ...
    # Return metrics dict
    return {"pearson_r": 0.0, "spearman_rho": 0.0, "rmse": 0.0} # Mock
