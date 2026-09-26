"""
abCAN-v2 — Module G: Prediction Head

Final MLP layers to predict the scalar ΔΔG value from the aggregated representation.

Input:  Aggregated representation (B, D)
Output: ΔΔG scalar (B, 1)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PredictionHead(nn.Module):
    """
    Module G: Prediction Head
    
    Standard MLP to map the final representation to a single ΔΔG prediction.
    
    Architecture (from diagram):
      - 128 → 64 → 16 → 1
      - Activations: ReLU
      - Dropout for regularization
    """
    
    def __init__(
        self,
        layers: List[int] = [128, 64, 16, 1],
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.layers_config = layers
        
        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU
        elif activation.lower() == "gelu":
            self.activation_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        modules = []
        
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i+1]
            
            modules.append(nn.Linear(in_dim, out_dim))
            
            # Add activation and dropout for all layers except the last one
            if i < len(layers) - 2:
                modules.append(self.activation_fn())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
                    
        self.mlp = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) aggregated feature vector.
            
        Returns:
            (B, 1) predicted ΔΔG.
        """
        return self.mlp(x)
