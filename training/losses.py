"""
abCAN-v2 — Custom Loss Functions

Combined loss from the architecture diagram:
  Loss = MSE + λ1 * Weighted_MSE + λ2 * (Cosine_Loss or Hinge_Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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


class CombinedLoss(nn.Module):
    """
    Combined Loss for abCAN-v2 training.
    
    Includes:
    1. Standard MSE
    2. Weighted MSE (replicate experimental noise weighting)
    3. Censored Hinge Loss for assay saturation
    """
    def __init__(
        self,
        mse_weight: float = 1.0,
        weighted_mse_weight: float = 0.5,
        hinge_weight: float = 0.3,
        saturation_margin: float = 8.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.weighted_mse_weight = weighted_mse_weight
        self.hinge_weight = hinge_weight
        
        self.hinge = CensoredHingeLoss(margin=saturation_margin)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, 1) or (B,) predicted ΔΔG.
            target: (B, 1) or (B,) true ΔΔG.
            sample_weights: (B,) experimental certainty weights.
            
        Returns:
            Dict containing 'loss' (total) and individual components.
        """
        pred = pred.view(-1)
        target = target.view(-1)
        sample_weights = sample_weights.view(-1)
        
        # 1. Standard MSE
        mse_loss = F.mse_loss(pred, target)
        
        # 2. Weighted MSE (incorporates experimental variance/replicates)
        # Weights are assumed to be normalized.
        weighted_mse = torch.mean(sample_weights * (pred - target).pow(2))
        
        # 3. Censored Hinge
        hinge_loss = self.hinge(pred, target)
        
        # Total
        total_loss = (
            self.mse_weight * mse_loss +
            self.weighted_mse_weight * weighted_mse +
            self.hinge_weight * hinge_loss
        )
        
        return {
            "loss": total_loss,
            "mse": mse_loss,
            "weighted_mse": weighted_mse,
            "hinge": hinge_loss,
        }
