"""
abCAN-v2 — Custom Loss Functions

Combined loss from the architecture diagram:
  Loss = MSE + λ1 * Weighted_MSE + λ2 * (Cosine_Loss or Hinge_Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CensoredHingeLoss(nn.Module):
    """
    Censored Hinge Loss for out-of-bounds ΔΔG values.
    
    Useful when experimental assays saturate (e.g., ΔΔG > 8 or < -8),
    penalizing predictions only if they fall on the wrong side of the threshold.
    """
    def __init__(self, margin: float = 8.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If target > margin, we only penalize if pred < margin
        upper_bound_mask = target >= self.margin
        upper_loss = F.relu(self.margin - pred[upper_bound_mask]).pow(2)
        
        # If target < -margin, we only penalize if pred > -margin
        lower_bound_mask = target <= -self.margin
        lower_loss = F.relu(pred[lower_bound_mask] - (-self.margin)).pow(2)
        
        loss = 0.0
        if len(upper_loss) > 0:
            loss = loss + upper_loss.mean()
        if len(lower_loss) > 0:
            loss = loss + lower_loss.mean()
            
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(0.0, device=pred.device)


class PairwiseAffinityRankingLoss(nn.Module):
    """
    Auxiliary Pairwise Affinity Ranking Loss.
    Penalizes incorrect ranking order of mutant pairs within the same batch/complex.
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Pairwise differences
        diff_target = target.unsqueeze(0) - target.unsqueeze(1) # (N, N)
        diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)       # (N, N)
        
        # We look at pairs where target_i > target_j
        mask = diff_target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)
            
        # Hinge loss: max(0, margin - (pred_i - pred_j))
        loss = F.relu(self.margin - diff_pred[mask])
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss corresponding to Stage 4 (Multi-Task Training):
    1. Primary ΔΔG Regression (MSE + Weighted MSE + Censored Hinge)
    2. Auxiliary Pairwise Affinity Ranking Loss
    3. Auxiliary Affinity Direction Binary Classification Loss
    """
    def __init__(
        self,
        mse_weight: float = 1.0,
        weighted_mse_weight: float = 0.5,
        hinge_weight: float = 0.3,
        ranking_weight: float = 0.4,
        aux_bce_weight: float = 0.3,
        saturation_margin: float = 8.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.weighted_mse_weight = weighted_mse_weight
        self.hinge_weight = hinge_weight
        self.ranking_weight = ranking_weight
        self.aux_bce_weight = aux_bce_weight
        
        self.hinge = CensoredHingeLoss(margin=saturation_margin)
        self.ranking = PairwiseAffinityRankingLoss(margin=0.1)
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        pred_ddg: torch.Tensor,
        target_ddg: torch.Tensor,
        sample_weights: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred = pred_ddg.view(-1)
        target = target_ddg.view(-1)
        weights = sample_weights.view(-1)
        
        # 1. Primary Regression Losses
        mse_loss = F.mse_loss(pred, target)
        weighted_mse = torch.mean(weights * (pred - target).pow(2))
        hinge_loss = self.hinge(pred, target)
        
        # 2. Auxiliary Pairwise Ranking Loss
        ranking_loss = self.ranking(pred, target)
        
        # 3. Auxiliary Binary Direction (Gain=1 if target > 0 else 0)
        aux_bce_loss = torch.tensor(0.0, device=pred.device)
        if aux_logits is not None:
            direction_target = (target > 0).float().view(-1, 1)
            aux_bce_loss = self.bce(aux_logits, direction_target)
            
        total_loss = (
            self.mse_weight * mse_loss +
            self.weighted_mse_weight * weighted_mse +
            self.hinge_weight * hinge_loss +
            self.ranking_weight * ranking_loss +
            self.aux_bce_weight * aux_bce_loss
        )
        
        return {
            "loss": total_loss,
            "mse": mse_loss,
            "weighted_mse": weighted_mse,
            "hinge": hinge_loss,
            "ranking": ranking_loss,
            "aux_bce": aux_bce_loss,
        }

