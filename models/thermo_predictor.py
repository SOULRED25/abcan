"""
Anti-Symmetric Thermodynamic Predictor

Architecture:
  - Paired Difference Vector: h_fused(Mut) - h_fused(WT)
  - Direct ΔLLR Prior Injection
  - Siamese Prediction MLP
    -> Predicted ΔΔG (kcal/mol)
    -> Auxiliary Affinity Direction (Gain vs Loss)
"""

import torch
import torch.nn as nn
from typing import Tuple, List

class SiamesePredictionMLP(nn.Module):
    """
    Siamese MLP that outputs both the continuous scalar and an auxiliary class.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.shared_mlp = nn.Sequential(*layers)
        
        # Primary Output: Continuous ΔΔG
        self.ddg_head = nn.Linear(curr_dim, 1)
        
        # Auxiliary Output: Affinity Direction (Gain=1, Loss=0)
        self.auxiliary_head = nn.Linear(curr_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_mlp(x)
        
        pred_ddg = self.ddg_head(features)
        
        # Logits for binary classification (BCEWithLogitsLoss)
        aux_logits = self.auxiliary_head(features)
        
        return pred_ddg, aux_logits


class AntiSymmetricPredictor(nn.Module):
    """
    Anti-Symmetric Thermodynamic Predictor.
    
    1. Computes difference: Mut_fused - WT_fused
    2. Injects Zero-Shot ΔLLR prior.
    3. Feeds into Siamese MLP.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        mlp_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Aggregation mechanism for the sequence/structure length. 
        # Mean pooling over mutated residues is common before the final MLP.
        
        # Input dim = hidden_dim (from difference vector) + 1 (for ΔLLR scalar prior)
        self.siamese_mlp = SiamesePredictionMLP(
            input_dim=hidden_dim + 1,
            hidden_layers=mlp_layers,
            dropout=dropout
        )
        
    def forward(
        self,
        h_fused_wt: torch.Tensor,
        h_fused_mut: torch.Tensor,
        mutation_mask: torch.Tensor,
        zero_shot_llr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_fused_wt: (B, N, D)
            h_fused_mut: (B, N, D)
            mutation_mask: (B, N) boolean mask.
            zero_shot_llr: (B, 1) external prior computed via ESM2 log-likelihoods.
            
        Returns:
            pred_ddg: (B, 1)
            aux_logits: (B, 1)
        """
        B, N, D = h_fused_wt.shape
        
        # Paired Difference Vector
        h_diff = h_fused_mut - h_fused_wt  # Strictly Anti-Symmetric (Mut - WT)
        
        # Pool features at the mutation site
        # To handle multiple mutations, we mean-pool across True indices in the mask
        pooled_diff = torch.zeros(B, D, device=h_diff.device)
        
        for i in range(B):
            mut_idx = torch.where(mutation_mask[i])[0]
            if len(mut_idx) > 0:
                pooled_diff[i] = h_diff[i, mut_idx].mean(dim=0)
            else:
                pooled_diff[i] = h_diff[i].mean(dim=0)
                
        # Direct ΔLLR Prior Injection
        # Concatenate the scalar ΔLLR directly into the final feature vector
        injected_features = torch.cat([pooled_diff, zero_shot_llr], dim=-1)  # (B, D + 1)
        
        # Predict
        pred_ddg, aux_logits = self.siamese_mlp(injected_features)
        
        return pred_ddg, aux_logits
