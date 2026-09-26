"""
abCAN-v2 — Main Model Architecture

Combines Modules A through G to form the full abCAN-v2 pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.mpnn_structural import MPNNStructuralEncoder
from models.mpnn_sequence import MPNNSequenceEncoderV2
from models.interface_attention import InterfaceAwareDualStreamAttention
from models.physicochemical import PhysicochemicalResidualInjection
from models.antisymmetric import AntisymmetricDifference
from models.aggregation import MutationAwareWeightedAggregation
from models.prediction_head import PredictionHead


class AbCANv2(nn.Module):
    """
    abCAN-v2 Full Architecture
    
    Predicts Antibody-Antigen ΔΔG by fusing structural, sequence, and
    biochemical prior branches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        model_cfg = config.get("model", {})
        
        # ─────────────────────────────────────────────
        # Stage 4A: MPNN Structural Encoder
        # ─────────────────────────────────────────────
        mod_a = model_cfg.get("mpnn_structural", {})
        self.structural_encoder = MPNNStructuralEncoder(
            input_dim=config.get("features", {}).get("structural", {}).get("pca_components", 3) * 3 + 
                      config.get("features", {}).get("structural", {}).get("positional_encoding_dim", 16) +
                      (4 if config.get("features", {}).get("structural", {}).get("quaternion_geometry", True) else 0) + 3, # e.g. 32
            hidden_dim=mod_a.get("hidden_dim", 128),
            num_layers=mod_a.get("num_layers", 3),
            dropout=mod_a.get("dropout", 0.1),
            residual_skip=mod_a.get("residual_skip", True),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4B: MPNN Sequence Encoder v2
        # ─────────────────────────────────────────────
        mod_b = model_cfg.get("mpnn_sequence", {})
        self.sequence_encoder = MPNNSequenceEncoderV2(
            esm2_dim=mod_b.get("esm2_dim", 1280),
            hidden_dim=mod_b.get("hidden_dim", 128),
            num_heads=mod_b.get("num_heads", 4),
            cross_attention=mod_b.get("cross_attention", True),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4C: Interface-Aware Dual-Stream Attention
        # ─────────────────────────────────────────────
        mod_c = model_cfg.get("interface_attention", {})
        self.interface_attention = InterfaceAwareDualStreamAttention(
            hidden_dim=mod_c.get("hidden_dim", 128),
            num_heads=mod_c.get("num_heads", 8),
            cross_attention=mod_c.get("cross_attention", True),
            interface_zone_gating=mod_c.get("interface_zone_gating", True),
            dropout=mod_c.get("dropout", 0.1),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4D: Physicochemical Residual Injection
        # ─────────────────────────────────────────────
        mod_d = model_cfg.get("physicochemical", {})
        self.physicochemical_injection = PhysicochemicalResidualInjection(
            input_dim=mod_d.get("input_dim", 6),
            hidden_dim=mod_d.get("hidden_dim", 64),
            output_dim=mod_d.get("output_dim", 128),
            residual_gate=mod_d.get("residual_gate", True),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4E: Antisymmetric Difference
        # ─────────────────────────────────────────────
        mod_e = model_cfg.get("antisymmetric", {})
        self.antisymmetric_diff = AntisymmetricDifference(
            hidden_dim=128,  # typically fixed to match main dimension
            mode=mod_e.get("mode", "subtract"),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4F: Mutation-Aware Weighted Aggregation
        # ─────────────────────────────────────────────
        mod_f = model_cfg.get("aggregation", {})
        self.aggregation = MutationAwareWeightedAggregation(
            hidden_dim=mod_f.get("hidden_dim", 128),
            num_heads=mod_f.get("num_heads", 4),
            attention_weighted=mod_f.get("attention_weighted", True),
        )
        
        # ─────────────────────────────────────────────
        # Stage 4G: Prediction Head
        # ─────────────────────────────────────────────
        mod_g = model_cfg.get("prediction_head", {})
        self.prediction_head = PredictionHead(
            layers=mod_g.get("layers", [128, 64, 16, 1]),
            activation=mod_g.get("activation", "relu"),
            dropout=mod_g.get("dropout", 0.2),
        )
        
    def forward(
        self,
        struct_node_feat: torch.Tensor,
        struct_edge_index: torch.Tensor,
        struct_edge_attr: torch.Tensor,
        wt_seq_emb: torch.Tensor,
        mut_seq_emb: torch.Tensor,
        biochem_features: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a batched or single graph.
        
        Since graphs vary in size, this is typically called per-graph or using 
        PyTorch Geometric's batching mechanism where N is total nodes in batch.
        
        Assuming a single complex (or a PyG batch viewed appropriately):
        
        Args:
            struct_node_feat: (B, N, D_struct)
            struct_edge_index: (2, E)
            struct_edge_attr: (E, D_edge)
            wt_seq_emb: (B, N, 1280)
            mut_seq_emb: (B, N, 1280)
            biochem_features: (B, N, 6)
            interface_mask: (B, N) bool
            mutation_mask: (B, N) bool
            padding_mask: (B, N) bool (True for valid)
            
        Returns:
            (B, 1) predicted ΔΔG
        """
        
        # We need to handle batched operations slightly differently if using PyG Data objects vs standard tensors.
        # This implementation assumes inputs are padded standard tensors (B, N, ...) or we loop if B=1.
        # For simplicity, assuming batched tensors (B, N, ...) here.
        B, N, _ = struct_node_feat.shape
        
        # 1. Structural Encoder
        # PyG operations expect flat node features (B*N, D), so we reshape
        flat_struct_node = struct_node_feat.view(-1, struct_node_feat.size(-1))
        
        # (Assuming edge_index is appropriately shifted for batched PyG, 
        # or B=1 in standard implementation)
        struct_rep = self.structural_encoder(flat_struct_node, struct_edge_index, struct_edge_attr)
        struct_rep = struct_rep.view(B, N, -1)  # (B, N, hidden_dim)
        
        # 2. Sequence Encoder
        wt_seq_rep, mut_seq_rep = self.sequence_encoder(
            wt_seq_emb, mut_seq_emb, padding_mask
        )
        
        # 3. Interface-Aware Dual-Stream Attention
        wt_fused, mut_fused = self.interface_attention(
            struct_rep, wt_seq_rep, mut_seq_rep, interface_mask, padding_mask
        )
        
        # 4. Physicochemical Residual Injection
        # Inject biochem features into both WT and Mut streams (or just Mut depending on design; 
        # normally biochem features represent the change Mut-WT, so injecting it into Mut stream or both)
        # We inject into both to provide context, or just into Mut. Let's inject into both here.
        wt_fused = self.physicochemical_injection(wt_fused, biochem_features)
        mut_fused = self.physicochemical_injection(mut_fused, biochem_features)
        
        # 5. Antisymmetric Difference
        diff_rep = self.antisymmetric_diff(wt_fused, mut_fused)
        
        # 6. Aggregation
        agg_rep = self.aggregation(diff_rep, mutation_mask, padding_mask)
        
        # 7. Prediction Head
        pred_ddg = self.prediction_head(agg_rep)
        
        return pred_ddg
