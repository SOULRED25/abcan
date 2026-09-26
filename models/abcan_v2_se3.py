"""
Main Architecture Class for abCAN-v2 (SE(3) Dual-Stream)

1. Dual-Stream Encoders (EGNN + ESM2)
2. Bi-Directional Cross-Attention Engine
3. Anti-Symmetric Thermodynamic Predictor
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from models.se3_egnn import SE3_EGNN
from models.cross_attention import BiDirectionalCrossAttention
from models.thermo_predictor import AntiSymmetricPredictor


class AbCANv2_SE3(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        cross_cfg = config.get("cross_attention", {})
        enc_cfg = config.get("encoders", {})
        pred_cfg = config.get("predictor", {})
        
        hidden_dim = cross_cfg.get("hidden_dim", 128)
        
        # ─────────────────────────────────────────────
        # Stream 1: Structure (SE(3)-EGNN)
        # ─────────────────────────────────────────────
        egnn_cfg = enc_cfg.get("se3_egnn", {})
        # Note: adjust node_in_dim and edge_in_dim based on preprocessing
        self.struct_encoder = SE3_EGNN(
            node_in_dim=32,   # Example: initial scalar features
            edge_in_dim=1,    # Example: distance / bond features
            hidden_dim=hidden_dim,
            num_layers=egnn_cfg.get("num_layers", 4)
        )
        
        # ─────────────────────────────────────────────
        # Stream 2: Sequence (ESM2 Projection)
        # ─────────────────────────────────────────────
        esm_cfg = enc_cfg.get("esm2", {})
        self.seq_proj = nn.Linear(esm_cfg.get("embedding_dim", 1280), hidden_dim)
        self.seq_norm = nn.LayerNorm(hidden_dim)
        
        # ─────────────────────────────────────────────
        # Engine: Bi-Directional Cross-Attention
        # ─────────────────────────────────────────────
        biases = cross_cfg.get("biases", {})
        self.cross_attention = BiDirectionalCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=cross_cfg.get("num_heads", 8),
            dropout=cross_cfg.get("dropout", 0.1),
            lambda_iface=biases.get("lambda_iface", 2.0),
            lambda_mut=biases.get("lambda_mut", 5.0)
        )
        
        # ─────────────────────────────────────────────
        # Predictor: Anti-Symmetric Siamese MLP
        # ─────────────────────────────────────────────
        mlp_cfg = pred_cfg.get("siamese_mlp", {})
        self.predictor = AntiSymmetricPredictor(
            hidden_dim=hidden_dim,
            mlp_layers=mlp_cfg.get("layers", [128, 64, 32]),
            dropout=mlp_cfg.get("dropout", 0.2)
        )

    def forward(
        self,
        struct_h: torch.Tensor,
        struct_x: torch.Tensor,
        struct_edge_index: torch.Tensor,
        struct_edge_attr: torch.Tensor,
        seq_wt: torch.Tensor,
        seq_mut: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
        zero_shot_llr: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the full Dual-Stream SE(3) architecture.
        
        Args:
            struct_h: Node scalar features (N, node_in_dim)
            struct_x: 3D coordinates (N, 3)
            struct_edge_index: (2, E)
            struct_edge_attr: (E, edge_in_dim)
            seq_wt: ESM2 embeddings for wild-type (B, N, 1280)
            seq_mut: ESM2 embeddings for mutant (B, N, 1280)
            interface_mask: (B, N) boolean mask
            mutation_mask: (B, N) boolean mask
            zero_shot_llr: (B, 1) ESM2 Zero-Shot LLR Prior
            padding_mask: (B, N)
            
        Returns:
            pred_ddg: (B, 1)
            aux_logits: (B, 1)
        """
        B, N, _ = seq_wt.shape
        
        # 1. Structure Stream (Shared for WT and Mut since backbone is identical)
        h_struct_invariant = self.struct_encoder(
            h=struct_h, 
            x=struct_x, 
            edge_index=struct_edge_index, 
            edge_attr=struct_edge_attr
        )
        # Reshape flat PyG output to batched shape
        h_struct_invariant = h_struct_invariant.view(B, N, -1)
        
        # 2. Sequence Stream
        h_seq_wt = self.seq_norm(self.seq_proj(seq_wt))
        h_seq_mut = self.seq_norm(self.seq_proj(seq_mut))
        
        # 3. Bi-Directional Cross-Attention Engine (Run for both WT and Mut)
        h_fused_wt = self.cross_attention(
            h_struct=h_struct_invariant,
            h_seq=h_seq_wt,
            interface_mask=interface_mask,
            mutation_mask=mutation_mask,
            padding_mask=padding_mask
        )
        
        h_fused_mut = self.cross_attention(
            h_struct=h_struct_invariant,
            h_seq=h_seq_mut,
            interface_mask=interface_mask,
            mutation_mask=mutation_mask,
            padding_mask=padding_mask
        )
        
        # 4. Anti-Symmetric Thermodynamic Predictor
        pred_ddg, aux_logits = self.predictor(
            h_fused_wt=h_fused_wt,
            h_fused_mut=h_fused_mut,
            mutation_mask=mutation_mask,
            zero_shot_llr=zero_shot_llr
        )
        
        return pred_ddg, aux_logits
