"""
Dataset for abCAN training and the held-out M515 benchmark.

The cache contains real, precomputed model inputs. This loader deliberately
rejects the old synthetic examples so they cannot produce misleading scores.
"""
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from config import resolve_path

_CACHE_BY_PATH: Dict[Path, List[Dict[str, Any]]] = {}
_REQUIRED_FIELDS = {
    "split",
    "struct_h",
    "struct_x",
    "struct_edge_index",
    "struct_edge_attr",
    "seq_wt",
    "seq_mut",
    "zero_shot_llr",
    "interface_mask",
    "mutation_mask",
    "ddg",
}


def _load_feature_cache(path: Path) -> List[Dict[str, Any]]:
    if path not in _CACHE_BY_PATH:
        if not path.is_file():
            raise FileNotFoundError(
                f"abCAN M515 feature cache not found: {path}. "
                "The original M515 dataset and real feature cache are required; "
                "synthetic examples are disabled."
            )
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded.get("samples")
        if not isinstance(loaded, list):
            raise ValueError("M515 feature cache must be a list of sample dictionaries.")
        for index, sample in enumerate(loaded):
            if not isinstance(sample, dict):
                raise ValueError(f"M515 cache sample {index} is not a dictionary.")
            missing = _REQUIRED_FIELDS - set(sample)
            if missing:
                raise ValueError(
                    f"M515 cache sample {index} is missing fields: {sorted(missing)}"
                )
        _CACHE_BY_PATH[path] = loaded
    return _CACHE_BY_PATH[path]


class AbCANDatasetSE3(Dataset):
    """Selects train, validation, or the held-out m515_test split.

    M515 is reserved for final scoring and must not be used for training,
    checkpoint selection, or hyperparameter tuning.
    """

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        dataset_cfg = config.get("dataset", {})
        active = dataset_cfg.get("active_benchmark")
        if active != "abcan_m515":
            raise ValueError(
                f"This training path only supports abcan_m515; configured active "
                f"benchmark is {active!r}."
            )

        benchmark_cfg = dataset_cfg.get("abcan_m515", {})
        cache_value = benchmark_cfg.get("feature_cache")
        if not cache_value:
            raise ValueError("dataset.abcan_m515.feature_cache is not configured.")

        self.split = split
        self.cache_path = resolve_path(cache_value).resolve()
        all_samples = _load_feature_cache(self.cache_path)
        self.data = [
            sample for sample in all_samples
            if str(sample["split"]).lower() == split.lower()
        ]
        if not self.data:
            raise ValueError(
                f"No abCAN M515 samples found for split {split!r} in {self.cache_path}."
            )
        self._validate_samples()

    def _validate_samples(self) -> None:
        for index, sample in enumerate(self.data):
            n_nodes = sample["struct_h"].shape[0]
            if sample["struct_x"].shape != (n_nodes, 3):
                raise ValueError(f"M515 sample {index}: struct_x must have shape (N, 3).")
            if sample["seq_wt"].shape[0] != n_nodes or sample["seq_mut"].shape[0] != n_nodes:
                raise ValueError(
                    f"M515 sample {index}: structure and sequence residue counts differ."
                )
            if sample["interface_mask"].numel() != n_nodes:
                raise ValueError(f"M515 sample {index}: invalid interface_mask length.")
            if sample["mutation_mask"].numel() != n_nodes:
                raise ValueError(f"M515 sample {index}: invalid mutation_mask length.")
            if sample["struct_edge_index"].ndim != 2 or sample["struct_edge_index"].shape[0] != 2:
                raise ValueError(
                    f"M515 sample {index}: struct_edge_index must have shape (2, E)."
                )
            if sample["struct_edge_attr"].shape[0] != sample["struct_edge_index"].shape[1]:
                raise ValueError(
                    f"M515 sample {index}: edge indices and edge features differ in length."
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = dict(self.data[index])
        sample.setdefault(
            "padding_mask",
            torch.ones(sample["struct_h"].shape[0], dtype=torch.bool),
        )
        sample.setdefault("sample_weight", torch.tensor(1.0, dtype=torch.float32))
        return sample


def collate_abcan_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pads residue tensors and offsets graph edges for a mixed-size batch."""
    if not samples:
        raise ValueError("Cannot collate an empty abCAN batch.")

    max_nodes = max(sample["struct_h"].shape[0] for sample in samples)
    batched: Dict[str, List[torch.Tensor]] = {
        key: [] for key in (
            "struct_h", "struct_x", "seq_wt", "seq_mut",
            "interface_mask", "mutation_mask", "padding_mask",
        )
    }
    edge_indices = []
    edge_attrs = []

    for graph_index, sample in enumerate(samples):
        n_nodes = sample["struct_h"].shape[0]
        pad_nodes = max_nodes - n_nodes

        for key in ("struct_h", "struct_x", "seq_wt", "seq_mut"):
            value = sample[key]
            padding = value.new_zeros((pad_nodes, *value.shape[1:]))
            batched[key].append(torch.cat((value, padding), dim=0))

        for key in ("interface_mask", "mutation_mask"):
            value = sample[key].reshape(-1).bool()
            batched[key].append(torch.cat((value, value.new_zeros(pad_nodes)), dim=0))

        valid_mask = sample["padding_mask"].reshape(-1).bool()
        batched["padding_mask"].append(
            torch.cat((valid_mask, valid_mask.new_zeros(pad_nodes)), dim=0)
        )

        edge_indices.append(sample["struct_edge_index"].long() + graph_index * max_nodes)
        edge_attrs.append(sample["struct_edge_attr"])

    result = {key: torch.stack(values, dim=0) for key, values in batched.items()}
    result["struct_h"] = result["struct_h"].reshape(-1, result["struct_h"].shape[-1])
    result["struct_x"] = result["struct_x"].reshape(-1, 3)
    result["struct_edge_index"] = torch.cat(edge_indices, dim=1)
    result["struct_edge_attr"] = torch.cat(edge_attrs, dim=0)
    result["zero_shot_llr"] = torch.stack(
        [sample["zero_shot_llr"].reshape(1) for sample in samples], dim=0
    )
    result["ddg"] = torch.stack(
        [sample["ddg"].reshape(1).float() for sample in samples], dim=0
    )
    result["sample_weight"] = torch.stack(
        [sample["sample_weight"].reshape(1).float() for sample in samples], dim=0
    )
    return result
