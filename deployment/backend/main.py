"""FastAPI status API for the active abCAN M515 benchmark."""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_config, resolve_path

CONFIG = load_config()
DATASET_CONFIG = CONFIG.get("dataset", {})
BENCHMARK_CONFIG = DATASET_CONFIG.get("abcan_m515", {})
RESULTS_PATH = resolve_path(
    CONFIG.get("logging", {}).get(
        "results_path", "evaluation/results/abcan_m515.json"
    )
)

PUBLISHED_REFERENCE = {
    "benchmark": "M515",
    "rmse_kcal_mol": 1.460,
    "pcc": 0.731,
    "table_rmse_kcal_mol": 1.476,
    "source": "https://academic.oup.com/bib/article/26/5/bbaf464/8251565",
    "note": (
        "The paper's abstract and results text report RMSE 1.460; its comparison "
        "table lists 1.476. PCC is 0.731 in both."
    ),
}

app = FastAPI(
    title="abCAN M515 Benchmark API",
    description="Status and metrics for the active abCAN M515 benchmark.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MutationRequest(BaseModel):
    chain: str
    resnum: int
    wild_aa: str
    mut_aa: str


class BatchPredictionRequest(BaseModel):
    mutations: List[MutationRequest]


def _path_exists(config_key: str) -> bool:
    value = BENCHMARK_CONFIG.get(config_key)
    return bool(value and resolve_path(value).is_file())


def _load_current_result() -> Optional[dict]:
    if not RESULTS_PATH.is_file():
        return None
    try:
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("benchmark") != "abcan_m515" or result.get("split") != "test":
        return None
    return result


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_benchmark": "abcan_m515",
        "training_manifest_available": _path_exists("training_manifest"),
        "m515_test_manifest_available": _path_exists("m515_test_manifest"),
        "feature_cache_available": _path_exists("feature_cache"),
        "project_test_score_available": _load_current_result() is not None,
        "prediction_ready": False,
    }


@app.get("/benchmark")
def benchmark_status():
    current_result = _load_current_result()
    training_available = _path_exists("training_manifest")
    m515_test_available = _path_exists("m515_test_manifest")
    features_available = _path_exists("feature_cache")
    return {
        "active_benchmark": {
            "id": "abcan_m515",
            "name": "abCAN M515",
            "target": "Antibody–antigen ΔΔG (kcal/mol)",
            "training_manifest_available": training_available,
            "m515_test_manifest_available": m515_test_available,
            "feature_cache_available": features_available,
        },
        "published_reference": PUBLISHED_REFERENCE,
        "current_project_result": current_result,
        "project_status": (
            "evaluated"
            if current_result
            else "awaiting_original_abcan_training_data"
            if not training_available
            else "awaiting_m515_test_data"
            if not m515_test_available
            else "awaiting_abcan_feature_cache"
            if not features_available
            else "ready_to_train"
        ),
        "future_datasets": [
            {"id": "abagym", "status": "coming_soon"},
            {"id": "sabdab2", "status": "coming_soon"},
        ],
        "future_approaches": [
            {"id": "sequence_only", "status": "coming_soon"},
            {"id": "structure_only", "status": "coming_soon"},
            {"id": "ensemble", "status": "coming_soon"},
        ],
    }


@app.post("/predict/single")
async def predict_single(
    pdb_file: UploadFile = File(...),
    chain: str = Form(...),
    resnum: int = Form(...),
    wild_aa: str = Form(...),
    mut_aa: str = Form(...),
):
    """Prediction remains disabled until a real abCAN checkpoint is available."""
    del pdb_file, chain, resnum, wild_aa, mut_aa
    raise HTTPException(
        status_code=503,
        detail=(
            "The abCAN PDB-to-feature prediction pipeline is not available yet. "
            "The API will not return a placeholder prediction."
        ),
    )


@app.post("/predict/batch")
async def predict_batch(
    pdb_file: UploadFile = File(...),
    mutations: Optional[str] = Form(default=None),
):
    """Batch prediction remains disabled until the active model is trained."""
    del pdb_file, mutations
    raise HTTPException(
        status_code=503,
        detail="Batch prediction is unavailable until the abCAN M515 model is trained.",
    )
