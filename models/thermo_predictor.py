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


class SharedEnergyEncoder(nn.Module):
    """
    Shared Energy Encoder predicting absolute thermodynamic state energy E(state).
    Guarantees path independence and strict thermodynamic anti-symmetry:
    DeltaDeltaG = E(Mut) - E(WT) + alpha * DeltaLR
    """
    def __init__(self, hidden_dim: int, mlp_layers: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        curr_dim = hidden_dim
        for h_dim in mlp_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        self.energy_mlp = nn.Sequential(*layers)
        self.energy_head = nn.Linear(curr_dim, 1)
        
    def forward(self, h_fused: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes scalar state energy E for a given state representation.
        """
        B, N, D = h_fused.shape
        pooled = torch.zeros(B, D, device=h_fused.device)
        for i in range(B):
            idx = torch.where(mask[i])[0]
            if len(idx) > 0:
                pooled[i] = h_fused[i, idx].mean(dim=0)
            else:
                pooled[i] = h_fused[i].mean(dim=0)
                
        feat = self.energy_mlp(pooled)
        energy = self.energy_head(feat)
        return energy, feat


class AntiSymmetricPredictor(nn.Module):
    """
    Anti-Symmetric Multi-Task Predictor based on Shared Energy State Decomposition.
    E(WT), E(Mut) -> DeltaDeltaG = E(Mut) - E(WT) + alpha * DeltaLR
    Also predicts auxiliary affinity gain/loss direction and pairwise ranking.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        mlp_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        self.shared_energy_encoder = SharedEnergyEncoder(
            hidden_dim=hidden_dim,
            mlp_layers=mlp_layers,
            dropout=dropout
        )
        # Learnable scaling coefficient for zero-shot DeltaLR
        self.alpha_lr = nn.Parameter(torch.tensor(0.1))
        
        # Auxiliary Head: Affinity Direction Logits (Gain vs Loss)
        self.auxiliary_direction_head = nn.Linear(mlp_layers[-1], 1)
        
    def forward(
        self,
        h_fused_wt: torch.Tensor,
        h_fused_mut: torch.Tensor,
        mutation_mask: torch.Tensor,
        zero_shot_llr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_fused_wt: (B, N, D) WT fused representations
            h_fused_mut: (B, N, D) Mutant fused representations
            mutation_mask: (B, N) boolean mask
            zero_shot_llr: (B, 1) ESM2 Zero-Shot LLR Prior
            
        Returns:
            pred_ddg: (B, 1) Predicted ΔΔG
            aux_logits: (B, 1) Affinity direction classification logits
            e_wt: (B, 1) WT absolute energy
            e_mut: (B, 1) Mutant absolute energy
        """
        e_wt, feat_wt = self.shared_energy_encoder(h_fused_wt, mutation_mask)
        e_mut, feat_mut = self.shared_energy_encoder(h_fused_mut, mutation_mask)
        
        # Exact Thermodynamic Energy Difference + Sequence LLR Contribution
        pred_ddg = (e_mut - e_wt) + self.alpha_lr * zero_shot_llr
        
        # Feature difference for auxiliary direction classification
        feat_diff = feat_mut - feat_wt
        aux_logits = self.auxiliary_direction_head(feat_diff)
        
        return pred_ddg, aux_logits, e_wt, e_mut

