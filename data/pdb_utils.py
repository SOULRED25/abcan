"""
abCAN-v2 — PDB Utilities
Handles PDB file parsing, residue extraction, coordinate retrieval,
and residue concordance matching (Exact / Offset / Not Found).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

try:
    from Bio.PDB import PDBParser, PPBuilder, NeighborSearch, Selection
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue as BioResidue
except ImportError:
    raise ImportError(
        "BioPython is required for PDB utilities. "
        "Install with: pip install biopython"
    )

logger = logging.getLogger(__name__)

# Standard amino acids (3-letter → 1-letter)
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ResidueInfo:
    """Information about a single residue in a PDB structure."""
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str           # 3-letter code
    aa_one_letter: str          # 1-letter code
    ca_coords: Optional[np.ndarray] = None    # Cα coordinates [x, y, z]
    all_atom_coords: Optional[Dict[str, np.ndarray]] = None
    b_factor: float = 0.0


@dataclass
class ConcordanceResult:
    """Result of PDB residue concordance matching."""
    status: str                  # "exact", "offset", "not_found"
    query_resnum: int
    matched_resnum: Optional[int] = None
    offset: int = 0
    chain_id: str = ""
    residue_name: str = ""


@dataclass
class ComplexStructure:
    """Parsed antibody-antigen complex structure."""
    pdb_id: str
    structure: Structure
    antibody_chains: List[str] = field(default_factory=list)
    antigen_chains: List[str] = field(default_factory=list)
    residues: Dict[str, List[ResidueInfo]] = field(default_factory=dict)
    

# ─────────────────────────────────────────────
# PDB Parser Wrapper
# ─────────────────────────────────────────────

class PDBHandler:
    """
    Handles PDB file parsing and residue-level operations 
    for antibody-antigen complexes.
    """
    
    def __init__(self, pdb_dir: Optional[str] = None, quiet: bool = True):
        """
        Args:
            pdb_dir: Directory containing PDB files.
            quiet: Suppress BioPython parser warnings.
        """
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.parser = PDBParser(QUIET=quiet)
        self._structure_cache: Dict[str, Structure] = {}
    
    def parse_pdb(self, pdb_path: Union[str, Path]) -> Structure:
        """
        Parse a PDB file and return the BioPython Structure object.
        
        Args:
            pdb_path: Path to PDB file.
            
        Returns:
            BioPython Structure object.
        """
        pdb_path = Path(pdb_path)
        pdb_id = pdb_path.stem.upper()
        
        if pdb_id in self._structure_cache:
            return self._structure_cache[pdb_id]
        
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        structure = self.parser.get_structure(pdb_id, str(pdb_path))
        self._structure_cache[pdb_id] = structure
        
        logger.info(f"Parsed PDB: {pdb_id} from {pdb_path}")
        return structure
    
    def load_complex(
        self,
        pdb_id: str,
        antibody_chains: Optional[List[str]] = None,
        antigen_chains: Optional[List[str]] = None,
    ) -> ComplexStructure:
        """
        Load an antibody-antigen complex from PDB.
        
        Args:
            pdb_id: PDB identifier (e.g., '1DVF').
            antibody_chains: Chain IDs for antibody (e.g., ['H', 'L']).
            antigen_chains: Chain IDs for antigen (e.g., ['A']).
            
        Returns:
            ComplexStructure with parsed residue information.
        """
        # Find the PDB file
        pdb_path = self._find_pdb_file(pdb_id)
        structure = self.parse_pdb(pdb_path)
        
        # Extract residues per chain
        residues = {}
        model = structure[0]  # Use first model
        
        for chain in model:
            chain_id = chain.get_id()
            chain_residues = self._extract_chain_residues(chain)
            if chain_residues:
                residues[chain_id] = chain_residues
        
        # Auto-detect chains if not specified
        if antibody_chains is None:
            antibody_chains = [c for c in residues.keys() if c in ('H', 'L')]
        if antigen_chains is None:
            antigen_chains = [c for c in residues.keys() if c not in ('H', 'L')]
        
        return ComplexStructure(
            pdb_id=pdb_id,
            structure=structure,
            antibody_chains=antibody_chains,
            antigen_chains=antigen_chains,
            residues=residues,
        )
    
    def _find_pdb_file(self, pdb_id: str) -> Path:
        """Locate PDB file by ID in the configured directory."""
        if self.pdb_dir is None:
            raise ValueError("pdb_dir not configured. Set it in the constructor.")
        
        pdb_id_lower = pdb_id.lower()
        
        # Try common naming conventions
        candidates = [
            self.pdb_dir / f"{pdb_id_lower}.pdb",
            self.pdb_dir / f"{pdb_id.upper()}.pdb",
            self.pdb_dir / f"pdb{pdb_id_lower}.ent",
            self.pdb_dir / f"{pdb_id_lower}.pdb.gz",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        raise FileNotFoundError(
            f"PDB file for {pdb_id} not found in {self.pdb_dir}. "
            f"Tried: {[c.name for c in candidates]}"
        )
    
    def _extract_chain_residues(self, chain: Chain) -> List[ResidueInfo]:
        """Extract all standard amino acid residues from a chain."""
        residues = []
        
        for residue in chain:
            # Skip water and heteroatoms (keep standard residues)
            hetflag = residue.get_id()[0]
            if hetflag != " ":
                continue
            
            resname = residue.get_resname().strip()
            if resname not in AA_3TO1:
                continue
            
            res_id = residue.get_id()
            resnum = res_id[1]
            insertion = res_id[2].strip()
            
            # Get Cα coordinates
            ca_coords = None
            if "CA" in residue:
                ca_coords = residue["CA"].get_vector().get_array().copy()
            
            # Get all atom coordinates
            all_atoms = {}
            b_factors = []
            for atom in residue:
                all_atoms[atom.get_name()] = atom.get_vector().get_array().copy()
                b_factors.append(atom.get_bfactor())
            
            residues.append(ResidueInfo(
                chain_id=chain.get_id(),
                residue_number=resnum,
                insertion_code=insertion,
                residue_name=resname,
                aa_one_letter=AA_3TO1[resname],
                ca_coords=ca_coords,
                all_atom_coords=all_atoms,
                b_factor=np.mean(b_factors) if b_factors else 0.0,
            ))
        
        return residues
    
    def get_ca_coordinates(self, complex_struct: ComplexStructure) -> Dict[str, np.ndarray]:
        """
        Get Cα coordinate matrices per chain.
        
        Returns:
            Dict mapping chain_id → (N, 3) array of Cα coordinates.
        """
        ca_coords = {}
        for chain_id, residues in complex_struct.residues.items():
            coords = []
            for res in residues:
                if res.ca_coords is not None:
                    coords.append(res.ca_coords)
                else:
                    logger.warning(
                        f"Missing Cα for {chain_id}:{res.residue_number} "
                        f"({res.residue_name})"
                    )
            if coords:
                ca_coords[chain_id] = np.stack(coords, axis=0)
        return ca_coords
    
    def get_sequence(self, complex_struct: ComplexStructure, chain_id: str) -> str:
        """Extract amino acid sequence for a chain."""
        if chain_id not in complex_struct.residues:
            raise ValueError(f"Chain {chain_id} not found in complex.")
        return "".join(r.aa_one_letter for r in complex_struct.residues[chain_id])


# ─────────────────────────────────────────────
# PDB Residue Concordance Matching
# ─────────────────────────────────────────────

class ResidueConcordanceMatcher:
    """
    Matches residue numbers from mutation data to PDB structure residues.
    
    Three matching modes (from architecture diagram):
      - Exact:     Direct residue number match
      - Offset:    Apply an integer offset to align numbering
      - Not Found: Residue cannot be located in the structure
    """
    
    def __init__(self, max_offset: int = 5, fallback: str = "skip"):
        """
        Args:
            max_offset: Maximum residue number offset to search.
            fallback: What to do when not found — "skip" or "impute".
        """
        self.max_offset = max_offset
        self.fallback = fallback
    
    def match_residue(
        self,
        query_chain: str,
        query_resnum: int,
        query_aa: str,
        complex_struct: ComplexStructure,
    ) -> ConcordanceResult:
        """
        Match a mutation residue to the PDB structure.
        
        Args:
            query_chain: Chain ID from mutation data.
            query_resnum: Residue number from mutation data.
            query_aa: Expected amino acid (1-letter code).
            complex_struct: Parsed complex structure.
            
        Returns:
            ConcordanceResult indicating match status.
        """
        if query_chain not in complex_struct.residues:
            logger.warning(f"Chain {query_chain} not found in {complex_struct.pdb_id}")
            return ConcordanceResult(
                status="not_found",
                query_resnum=query_resnum,
                chain_id=query_chain,
            )
        
        chain_residues = complex_struct.residues[query_chain]
        resnum_map = {r.residue_number: r for r in chain_residues}
        
        # 1. Try exact match
        if query_resnum in resnum_map:
            matched = resnum_map[query_resnum]
            if matched.aa_one_letter == query_aa:
                return ConcordanceResult(
                    status="exact",
                    query_resnum=query_resnum,
                    matched_resnum=query_resnum,
                    offset=0,
                    chain_id=query_chain,
                    residue_name=matched.residue_name,
                )
        
        # 2. Try offset matching
        for offset in range(-self.max_offset, self.max_offset + 1):
            if offset == 0:
                continue
            adjusted = query_resnum + offset
            if adjusted in resnum_map:
                matched = resnum_map[adjusted]
                if matched.aa_one_letter == query_aa:
                    logger.info(
                        f"Offset match: {query_chain}:{query_resnum} → "
                        f"{query_chain}:{adjusted} (offset={offset})"
                    )
                    return ConcordanceResult(
                        status="offset",
                        query_resnum=query_resnum,
                        matched_resnum=adjusted,
                        offset=offset,
                        chain_id=query_chain,
                        residue_name=matched.residue_name,
                    )
        
        # 3. Not found
        logger.warning(
            f"Concordance failed: {query_chain}:{query_resnum}{query_aa} "
            f"in {complex_struct.pdb_id}"
        )
        return ConcordanceResult(
            status="not_found",
            query_resnum=query_resnum,
            chain_id=query_chain,
        )
    
    def match_mutations(
        self,
        mutations: List[Dict],
        complex_struct: ComplexStructure,
    ) -> List[ConcordanceResult]:
        """
        Batch match mutations to PDB structure.
        
        Args:
            mutations: List of dicts with keys: chain, resnum, wild_aa.
            complex_struct: Parsed complex structure.
            
        Returns:
            List of ConcordanceResult for each mutation.
        """
        results = []
        stats = {"exact": 0, "offset": 0, "not_found": 0}
        
        for mut in mutations:
            result = self.match_residue(
                query_chain=mut["chain"],
                query_resnum=mut["resnum"],
                query_aa=mut["wild_aa"],
                complex_struct=complex_struct,
            )
            results.append(result)
            stats[result.status] += 1
        
        total = len(mutations)
        if total > 0:
            logger.info(
                f"Concordance results for {complex_struct.pdb_id}: "
                f"exact={stats['exact']}/{total} ({100*stats['exact']/total:.1f}%), "
                f"offset={stats['offset']}/{total} ({100*stats['offset']/total:.1f}%), "
                f"not_found={stats['not_found']}/{total} ({100*stats['not_found']/total:.1f}%)"
            )
        
        return results
