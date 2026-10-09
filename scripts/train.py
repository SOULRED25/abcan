"""Train the active abCAN source-only ΔΔG baseline.

This entry point deliberately uses only the normalized AB-Bind and SKEMPI
records in ``database/abcan/source_candidates_unfiltered.csv``. It does not
load SAbDab2, AbAgym, PROXiMATE, or M515. M515 remains unavailable locally,
so this script reports a grouped held-out source-set result, not an M515 score.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from config import load_config, resolve_path
from evaluation.metrics import calculate_metrics
from features.biochemical import get_blosum62_score, get_property_change

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_WITH_CHAIN = re.compile(r"([A-Za-z0-9]+):([A-Z])(\d+)([A-Z])")
_COMPACT_WITH_CHAIN = re.compile(r"([A-Za-z0-9])([A-Z])(\d+)([A-Z])")
_WITHOUT_CHAIN = re.compile(r"([A-Z])(\d+)([A-Z])")
_SOURCE_CODES = {"AB-Bind": 0.0, "SKEMPI 1": 1.0, "SKEMPI 2": 2.0}


def _mutations(value: object) -> Iterable[tuple[str, str, int, str]]:
    """Parse common AB-Bind/SKEMPI mutation forms without inventing residues."""
    for part in (item.strip() for item in str(value or "").upper().replace(";", ",").split(",")):
        match = _WITH_CHAIN.fullmatch(part)
        if match:
            chain, wt, position, mut = match.groups()
            yield chain, wt, int(position), mut
            continue
        match = _COMPACT_WITH_CHAIN.fullmatch(part)
        if match:
            chain, wt, position, mut = match.groups()
            yield chain, wt, int(position), mut
            continue
        match = _WITHOUT_CHAIN.fullmatch(part)
        if match:
            wt, position, mut = match.groups()
            yield "", wt, int(position), mut


def _feature_row(row: pd.Series) -> dict[str, float]:
    mutations = list(_mutations(row["mutation_cleaned"]))
    count = len(mutations)
    features = {
        "source_code": _SOURCE_CODES.get(str(row["source_dataset"]), -1.0),
        "mutation_count": float(count), "mean_position": 0.0, "mean_blosum62": 0.0,
        "sum_abs_polarity_change": 0.0, "sum_abs_charge_change": 0.0,
        "sum_abs_size_change": 0.0, "sum_abs_hydrophobicity_change": 0.0,
        "alanine_fraction": 0.0, "same_residue_fraction": 0.0,
    }
    if not mutations:
        return features
    positions, blosum = [], []
    alanine = unchanged = 0
    for _, wt, position, mut in mutations:
        positions.append(position)
        blosum.append(get_blosum62_score(wt, mut))
        features["sum_abs_polarity_change"] += abs(get_property_change(wt, mut, "polarity"))
        features["sum_abs_charge_change"] += abs(get_property_change(wt, mut, "charge"))
        features["sum_abs_size_change"] += abs(get_property_change(wt, mut, "size_normalized"))
        features["sum_abs_hydrophobicity_change"] += abs(get_property_change(wt, mut, "hydrophobicity_normalized"))
        alanine += int(mut == "A")
        unchanged += int(wt == mut)
    features["mean_position"] = float(np.mean(positions))
    features["mean_blosum62"] = float(np.mean(blosum))
    features["alanine_fraction"] = alanine / count
    features["same_residue_fraction"] = unchanged / count
    return features


def _grouped_splits(frame: pd.DataFrame, seed: int, validation_fraction: float, test_fraction: float):
    groups = frame["complex_id"].fillna("").astype(str)
    groups = groups.where(groups.ne(""), frame["source_record_id"].astype(str))
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_index, test_index = next(first.split(frame, groups=groups))
    train_val = frame.iloc[train_val_index]
    second = GroupShuffleSplit(n_splits=1, test_size=validation_fraction / (1.0 - test_fraction), random_state=seed + 1)
    train_local, validation_local = next(second.split(train_val, groups=groups.iloc[train_val_index]))
    return train_val.iloc[train_local], train_val.iloc[validation_local], frame.iloc[test_index]


def _score(name: str, model, frame: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(frame[feature_names])
    metrics = calculate_metrics(predictions, frame["ddg_kcal_mol"].to_numpy(dtype=float))
    result = {key: float(value) for key, value in metrics.items()}
    result["n"] = int(len(frame))
    logger.info("%s — RMSE %.4f kcal/mol — PCC %.4f (n=%d)", name, result["rmse"], result["pearson_r"], result["n"])
    return result, predictions


def main() -> None:
    config = load_config()
    source_cfg = config.get("dataset", {}).get("abcan_sources", {})
    manifest = resolve_path(source_cfg.get("candidate_manifest", "database/abcan/source_candidates_unfiltered.csv"))
    if not manifest.is_file():
        raise SystemExit(f"Cannot train: normalized abCAN source file is missing: {manifest}")
    frame = pd.read_csv(manifest)
    permitted = set(source_cfg.get("include_sources", _SOURCE_CODES))
    frame["ddg_kcal_mol"] = pd.to_numeric(frame["ddg_kcal_mol"], errors="coerce")
    frame = frame[frame["source_dataset"].isin(permitted) & frame["ddg_kcal_mol"].notna()].copy()
    if len(frame) < 30:
        raise SystemExit("Cannot train: fewer than 30 labeled AB-Bind/SKEMPI source records are available.")
    feature_frame = pd.DataFrame([_feature_row(row) for _, row in frame.iterrows()], index=frame.index)
    overlap = set(frame.columns) & set(feature_frame.columns)
    if overlap:
        frame = frame.drop(columns=list(overlap))
    frame = pd.concat((frame, feature_frame), axis=1)
    feature_names = list(feature_frame.columns)
    seed = int(config.get("project", {}).get("seed", 42))
    train, validation, test = _grouped_splits(frame, seed, float(source_cfg.get("validation_fraction", 0.15)), float(source_cfg.get("test_fraction", 0.15)))
    model = HistGradientBoostingRegressor(
        learning_rate=float(config.get("training", {}).get("learning_rate", 0.05)),
        max_iter=int(config.get("training", {}).get("max_iterations", 300)),
        max_leaf_nodes=int(config.get("training", {}).get("max_leaf_nodes", 31)),
        l2_regularization=float(config.get("training", {}).get("l2_regularization", 1.0)),
        random_state=seed,
    )
    model.fit(train[feature_names], train["ddg_kcal_mol"])
    logger.info("Training only on AB-Bind/SKEMPI sources: train=%d, validation=%d, test=%d", len(train), len(validation), len(test))
    validation_metrics, _ = _score("Validation", model, validation, feature_names)
    test_metrics, test_predictions = _score("Held-out source test", model, test, feature_names)
    logging_cfg = config.get("logging", {})
    checkpoint = resolve_path(logging_cfg.get("source_model", "checkpoints/abcan_source_baseline.joblib"))
    result_path = resolve_path(logging_cfg.get("source_results", "evaluation/results/abcan_source_only.json"))
    predictions_path = resolve_path(logging_cfg.get("source_predictions", "evaluation/results/abcan_source_only_predictions.csv"))
    for path in (checkpoint, result_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "feature_names": feature_names, "training_sources": sorted(permitted)}, checkpoint)
    result = {
        "experiment": "abcan_source_only_mutation_baseline", "input_manifest": str(manifest),
        "training_sources": sorted(permitted), "split_method": "grouped by complex_id; source-only split, not M515",
        "m515_evaluated": False, "validation": validation_metrics, "held_out_source_test": test_metrics, "model": str(checkpoint),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    test.loc[:, ["source_record_id", "source_dataset", "complex_id", "mutation_cleaned", "ddg_kcal_mol"]].assign(predicted_ddg_kcal_mol=test_predictions).to_csv(predictions_path, index=False)
    logger.info("Saved model to %s", checkpoint)
    logger.info("Saved source-only metrics to %s", result_path)


if __name__ == "__main__":
    main()
