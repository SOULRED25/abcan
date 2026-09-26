"""
abCAN-v2 — Stage 3A: 3D Structural Feature Extraction

Extracts structural features from PDB complexes:
  - Residue-level contact maps
  - K-NN graph construction  
  - PCA / Positional / Quaternion geometry
  - ESM-2 structural embeddings
  - 3D structural feature vectors
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import knn_graph
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from data.pdb_utils import ComplexStructure, ResidueInfo

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Contact Map Builder
# ─────────────────────────────────────────────

class ContactMapBuilder:
    """
    Builds residue-level contact maps from Cα coordinates.
    
    A contact exists between residues i, j if:
      dist(Cα_i, Cα_j) < threshold (default 8.0 Å)
    """
    
    def __init__(self, threshold: float = 8.0):
        self.threshold = threshold
    
    def build(self, ca_coords: np.ndarray) -> np.ndarray:
        """
        Build contact map from Cα coordinates.
        
        Args:
            ca_coords: (N, 3) array of Cα positions.
            
        Returns:
            (N, N) binary contact map.
        """
        N = len(ca_coords)
        # Pairwise Euclidean distances
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # (N, N, 3)
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))       # (N, N)
        
        # Binary contact map
        contact_map = (dist_matrix < self.threshold).astype(np.float32)
        np.fill_diagonal(contact_map, 0)  # No self-contacts
        
        return contact_map
    
    def build_distance_map(self, ca_coords: np.ndarray) -> np.ndarray:
        """Return full pairwise distance matrix (N, N)."""
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))


# ─────────────────────────────────────────────
# KNN Graph Builder
# ─────────────────────────────────────────────

class KNNGraphBuilder:
    """
    Builds K-Nearest Neighbor graph from 3D coordinates.
    
    Produces PyTorch Geometric edge_index for MPNN consumption.
    """
    
    def __init__(self, k: int = 30):
        self.k = k
    
    def build(self, ca_coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build KNN graph.
        
        Args:
            ca_coords: (N, 3) Cα coordinates.
            
        Returns:
            (edge_index, edge_attr) where:
              - edge_index: (2, E) tensor of edges
              - edge_attr: (E, 1) tensor of distances
        """
        coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)
        
        if HAS_PYG:
            # Use PyG's optimized knn_graph
            edge_index = knn_graph(
                coords_tensor,
                k=min(self.k, len(ca_coords) - 1),
                loop=False,
            )
        else:
            # Fallback: manual KNN
            edge_index = self._manual_knn(coords_tensor)
        
        # Compute edge distances
        src, dst = edge_index[0], edge_index[1]
        edge_vectors = coords_tensor[dst] - coords_tensor[src]
        edge_distances = torch.norm(edge_vectors, dim=-1, keepdim=True)
        
        return edge_index, edge_distances
    
    def _manual_knn(self, coords: torch.Tensor) -> torch.Tensor:
        """Fallback KNN computation without PyG."""
        N = coords.shape[0]
        k = min(self.k, N - 1)
        
        # Pairwise distances
        dist_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        
        # Set self-distance to inf
        dist_matrix.fill_diagonal_(float("inf"))
        
        # Top-k nearest neighbors
        _, indices = dist_matrix.topk(k, largest=False, dim=-1)  # (N, k)
        
        # Build edge_index
        src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        
        return edge_index


# ─────────────────────────────────────────────
# Geometric Feature Extractor
# ─────────────────────────────────────────────

class GeometricFeatureExtractor:
    """
    Extracts geometric features for each residue:
      - PCA-based local coordinate frames
      - Positional encodings (sinusoidal)
      - Quaternion geometry (local orientation)
    """
    
    def __init__(
        self,
        pca_components: int = 3,
        positional_dim: int = 16,
        use_quaternion: bool = True,
    ):
        self.pca_components = pca_components
        self.positional_dim = positional_dim
        self.use_quaternion = use_quaternion
    
    def extract(self, ca_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract geometric features.
        
        Args:
            ca_coords: (N, 3) Cα coordinates.
            
        Returns:
            Dict of feature arrays.
        """
        features = {}
        N = len(ca_coords)
        
        # 1. PCA-based local coordinate frames
        features["pca_frames"] = self._compute_local_frames(ca_coords)
        
        # 2. Sinusoidal positional encodings
        features["positional_encoding"] = self._sinusoidal_encoding(N)
        
        # 3. Quaternion geometry
        if self.use_quaternion:
            features["quaternion"] = self._compute_quaternions(ca_coords)
        
        # 4. Relative positions (centered)
        centroid = ca_coords.mean(axis=0)
        features["relative_coords"] = ca_coords - centroid
        
        return features
    
    def _compute_local_frames(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute local coordinate frames using PCA on neighbor Cα positions.
        
        Returns:
            (N, 9) — flattened 3×3 rotation matrices.
        """
        N = len(coords)
        frames = np.zeros((N, 9), dtype=np.float32)
        
        for i in range(N):
            # Use neighboring residues (i-2 to i+2) for local frame
            start = max(0, i - 2)
            end = min(N, i + 3)
            local_coords = coords[start:end]
            
            if len(local_coords) < 3:
                frames[i] = np.eye(3).flatten()
                continue
            
            # Center coordinates
            centered = local_coords - local_coords.mean(axis=0)
            
            # PCA via SVD
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                frame = Vt[:3, :3]  # Top 3 principal components
                # Ensure right-handedness
                if np.linalg.det(frame) < 0:
                    frame[2] *= -1
                frames[i] = frame.flatten()
            except np.linalg.LinAlgError:
                frames[i] = np.eye(3).flatten()
        
        return frames
    
    def _sinusoidal_encoding(self, N: int) -> np.ndarray:
        """
        Sinusoidal positional encoding.
        
        Returns:
            (N, positional_dim) encoding.
        """
        positions = np.arange(N, dtype=np.float32)
        dim = self.positional_dim
        
        encoding = np.zeros((N, dim), dtype=np.float32)
        for d in range(dim):
            if d % 2 == 0:
                encoding[:, d] = np.sin(positions / (10000 ** (d / dim)))
            else:
                encoding[:, d] = np.cos(positions / (10000 ** ((d - 1) / dim)))
        
        return encoding
    
    def _compute_quaternions(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute quaternion representations of local backbone geometry.
        
        Uses consecutive Cα triplets to define orientations.
        
        Returns:
            (N, 4) quaternion array.
        """
        N = len(coords)
        quaternions = np.zeros((N, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0  # Default: identity quaternion
        
        for i in range(1, N - 1):
            # Vectors from Cα_{i-1} to Cα_i and Cα_i to Cα_{i+1}
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i + 1] - coords[i]
            
            # Normalize
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            
            v1 = v1 / n1
            v2 = v2 / n2
            
            # Rotation quaternion from v1 to v2
            quaternions[i] = self._rotation_quaternion(v1, v2)
        
        # Copy endpoints
        if N > 1:
            quaternions[0] = quaternions[1]
            quaternions[-1] = quaternions[-2]
        
        return quaternions
    
    @staticmethod
    def _rotation_quaternion(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute quaternion representing rotation from v1 to v2."""
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        # Handle parallel/anti-parallel vectors
        if np.linalg.norm(cross) < 1e-8:
            if dot > 0:
                return np.array([1, 0, 0, 0], dtype=np.float32)
            else:
                # 180° rotation about any perpendicular axis
                perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(v1, perp)
                axis = axis / np.linalg.norm(axis)
                return np.array([0, axis[0], axis[1], axis[2]], dtype=np.float32)
        
        w = 1 + dot
        q = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float32)
        q = q / np.linalg.norm(q)
        
        return q


# ─────────────────────────────────────────────
# Main Structural Feature Extractor
# ─────────────────────────────────────────────

class StructuralFeatureExtractor:
    """
    Main 3D Structural Branch feature extractor.
    
    Combines:
      - Contact maps
      - KNN graphs
      - PCA / Positional / Quaternion geometry
      - Per-residue structural features
      
    Output feeds into MPNN Structural Encoder (Stage 4A).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        struct_cfg = config.get("features", {}).get("structural", {})
        
        self.contact_builder = ContactMapBuilder(
            threshold=config.get("preprocessing", {}).get("structure", {}).get(
                "contact_threshold", 8.0
            )
        )
        
        self.knn_builder = KNNGraphBuilder(
            k=struct_cfg.get("knn_k", 30)
        )
        
        self.geo_extractor = GeometricFeatureExtractor(
            pca_components=struct_cfg.get("pca_components", 3),
            positional_dim=struct_cfg.get("positional_encoding_dim", 16),
            use_quaternion=struct_cfg.get("quaternion_geometry", True),
        )
    
    def extract(
        self,
        complex_struct: ComplexStructure,
        chain_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all structural features for a complex.
        
        Args:
            complex_struct: Parsed PDB complex.
            chain_ids: Chains to extract features for (default: all).
            
        Returns:
            Dict of tensors:
              - node_features: (N, D) per-residue feature vectors
              - edge_index: (2, E) KNN graph edges
              - edge_attr: (E, F) edge features (distances)
              - contact_map: (N, N) residue contact map
              - positions: (N, 3) Cα coordinates
        """
        if chain_ids is None:
            chain_ids = list(complex_struct.residues.keys())
        
        # Collect all Cα coordinates across selected chains
        all_coords = []
        chain_offsets = {}
        
        for chain_id in chain_ids:
            if chain_id not in complex_struct.residues:
                continue
            
            chain_offsets[chain_id] = len(all_coords)
            
            for res in complex_struct.residues[chain_id]:
                if res.ca_coords is not None:
                    all_coords.append(res.ca_coords)
        
        if not all_coords:
            raise ValueError(
                f"No Cα coordinates found in {complex_struct.pdb_id} "
                f"for chains {chain_ids}"
            )
        
        ca_coords = np.stack(all_coords, axis=0)  # (N, 3)
        N = len(ca_coords)
        
        # 1. Contact map
        contact_map = self.contact_builder.build(ca_coords)
        
        # 2. KNN graph
        edge_index, edge_distances = self.knn_builder.build(ca_coords)
        
        # 3. Geometric features
        geo_features = self.geo_extractor.extract(ca_coords)
        
        # 4. Concatenate per-residue features
        # [relative_coords(3) + pca_frames(9) + positional(16) + quaternion(4)] = 32
        feature_parts = [
            geo_features["relative_coords"],     # (N, 3)
            geo_features["pca_frames"],           # (N, 9)
            geo_features["positional_encoding"],  # (N, 16)
        ]
        if "quaternion" in geo_features:
            feature_parts.append(geo_features["quaternion"])  # (N, 4)
        
        node_features = np.concatenate(feature_parts, axis=-1)
        
        # Convert to tensors
        result = {
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_distances,
            "contact_map": torch.tensor(contact_map, dtype=torch.float32),
            "positions": torch.tensor(ca_coords, dtype=torch.float32),
            "chain_offsets": chain_offsets,
            "num_residues": N,
        }
        
        logger.debug(
            f"Structural features for {complex_struct.pdb_id}: "
            f"N={N}, node_feat={node_features.shape}, edges={edge_index.shape[1]}"
        )
        
        return result
    
    def extract_interface(
        self,
        structural_features: Dict[str, torch.Tensor],
        complex_struct: ComplexStructure,
        distance_threshold: float = 10.0,
    ) -> torch.Tensor:
        """
        Identify interface residues between antibody and antigen.
        
        Returns:
            (N,) boolean mask indicating interface residues.
        """
        positions = structural_features["positions"]  # (N, 3)
        chain_offsets = structural_features["chain_offsets"]
        
        ab_chains = complex_struct.antibody_chains
        ag_chains = complex_struct.antigen_chains
        
        # Get indices for antibody and antigen residues
        ab_indices = []
        ag_indices = []
        
        cumulative = 0
        for chain_id in chain_offsets:
            n_res = len(complex_struct.residues.get(chain_id, []))
            chain_indices = list(range(cumulative, cumulative + n_res))
            
            if chain_id in ab_chains:
                ab_indices.extend(chain_indices)
            elif chain_id in ag_chains:
                ag_indices.extend(chain_indices)
            
            cumulative += n_res
        
        # Compute cross-chain distances
        N = positions.shape[0]
        interface_mask = torch.zeros(N, dtype=torch.bool)
        
        if ab_indices and ag_indices:
            ab_coords = positions[ab_indices]
            ag_coords = positions[ag_indices]
            
            # Cross-distance matrix
            cross_dist = torch.cdist(ab_coords, ag_coords)  # (n_ab, n_ag)
            
            # Interface: any residue within threshold of the partner
            ab_at_interface = cross_dist.min(dim=1).values < distance_threshold
            ag_at_interface = cross_dist.min(dim=0).values < distance_threshold
            
            for i, idx in enumerate(ab_indices):
                interface_mask[idx] = ab_at_interface[i]
            for i, idx in enumerate(ag_indices):
                interface_mask[idx] = ag_at_interface[i]
        
        return interface_mask
