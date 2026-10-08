"""Train on abCAN training data and evaluate once on held-out M515."""
import logging
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, resolve_path
from data.dataset import AbCANDatasetSE3, collate_abcan_batch
from models.abcan_v2_se3 import AbCANv2_SE3
from training.loop import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    active_benchmark = dataset_cfg.get("active_benchmark")
    if active_benchmark != "abcan_m515":
        raise SystemExit(
            "This entry point is restricted to dataset.active_benchmark=abcan_m515."
        )

    benchmark_cfg = dataset_cfg.get("abcan_m515", {})
    required_paths = {
        "original abCAN training manifest": benchmark_cfg.get("training_manifest"),
        "held-out M515 test manifest": benchmark_cfg.get("m515_test_manifest"),
        "prepared abCAN feature cache": benchmark_cfg.get("feature_cache"),
    }
    missing = [
        f"{description}: {resolve_path(path)}"
        for description, path in required_paths.items()
        if not path or not resolve_path(path).is_file()
    ]
    if missing:
        logger.error(
            "Cannot train: the original abCAN training and held-out M515 inputs "
            "are not present. Other bundled datasets are intentionally not substituted."
        )
        for path_description in missing:
            logger.error("Missing %s", path_description)
        raise SystemExit(2)

    seed_everything(config.get("project", {}).get("seed", 42))

    split_names = {
        "train": benchmark_cfg.get("train_split", "train"),
        "validation": benchmark_cfg.get("validation_split", "validation"),
        "test": benchmark_cfg.get("test_split", "test"),
    }
    datasets = {
        name: AbCANDatasetSE3(config, split=split)
        for name, split in split_names.items()
    }

    batch_size = config.get("training", {}).get("batch_size", 16)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "collate_fn": collate_abcan_batch,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets["validation"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    logger.info(
        "abCAN samples — train: %d, validation: %d, held-out M515 test: %d",
        len(datasets["train"]),
        len(datasets["validation"]),
        len(datasets["test"]),
    )

    model = AbCANv2_SE3(config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
