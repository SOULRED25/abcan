"""Training loop for the active abCAN M515 benchmark."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import resolve_path
from evaluation.metrics import calculate_metrics
from training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop when validation RMSE has not improved for a fixed number of epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        improved = self.best_loss is None or val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


class Trainer:
    """Train on train, select by validation RMSE, and score test once at the end."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        test_loader=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        train_cfg = config.get("training", {})
        optimizer_cfg = train_cfg.get("optimizer", {})
        component_cfg = train_cfg.get("loss", {}).get("components", {})

        requested_device = config.get("project", {}).get("device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable; using CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.model = self.model.to(self.device)

        optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
        optimizer_class = AdamW if optimizer_type == "adamw" else Adam
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=optimizer_cfg.get("lr", 1e-4),
            weight_decay=optimizer_cfg.get("weight_decay", 1e-5),
            betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        )
        self.epochs = train_cfg.get("epochs", 100)
        scheduler_cfg = train_cfg.get("scheduler", {})
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_cfg.get("T_max", self.epochs),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        self.criterion = MultiTaskLoss(
            mse_weight=component_cfg.get("mse", 1.0),
            weighted_mse_weight=component_cfg.get("weighted_mse", 0.0),
            hinge_weight=component_cfg.get("cosine_hinge", 0.0),
            ranking_weight=component_cfg.get("ranking", 0.0),
            aux_bce_weight=component_cfg.get("aux_bce", 0.0),
        ).to(self.device)

        early_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=early_cfg.get("patience", 20),
            min_delta=early_cfg.get("min_delta", 0.001),
        )

        logging_cfg = config.get("logging", {})
        self.checkpoint_path = resolve_path(
            logging_cfg.get("best_checkpoint", "checkpoints/abcan_m515_best.pt")
        )
        self.results_path = resolve_path(
            logging_cfg.get("results_path", "evaluation/results/abcan_m515.json")
        )
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _predict(self, batch: Dict[str, torch.Tensor]):
        outputs = self.model(
            struct_h=batch["struct_h"],
            struct_x=batch["struct_x"],
            struct_edge_index=batch["struct_edge_index"],
            struct_edge_attr=batch["struct_edge_attr"],
            seq_wt=batch["seq_wt"],
            seq_mut=batch["seq_mut"],
            interface_mask=batch["interface_mask"],
            mutation_mask=batch["mutation_mask"],
            zero_shot_llr=batch["zero_shot_llr"],
            padding_mask=batch["padding_mask"],
        )
        pred_ddg, aux_logits = outputs[:2]
        return pred_ddg, aux_logits

    def train_epoch(self) -> float:
        self.model.train()
        loss_sum = 0.0
        example_count = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            pred, aux_logits = self._predict(batch)
            loss_parts = self.criterion(
                pred,
                batch["ddg"],
                batch["sample_weight"],
                aux_logits=aux_logits if self.criterion.aux_bce_weight else None,
            )
            loss = loss_parts["loss"]
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()

            size = batch["ddg"].shape[0]
            loss_sum += loss.item() * size
            example_count += size

        if example_count == 0:
            raise ValueError("The abCAN M515 training split is empty.")
        self.scheduler.step()
        return loss_sum / example_count

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        loss_sum = 0.0
        example_count = 0
        predictions = []
        targets = []

        for batch in loader:
            batch = self._move_batch(batch)
            pred, aux_logits = self._predict(batch)
            loss_parts = self.criterion(
                pred,
                batch["ddg"],
                batch["sample_weight"],
                aux_logits=aux_logits if self.criterion.aux_bce_weight else None,
            )
            size = batch["ddg"].shape[0]
            loss_sum += loss_parts["loss"].item() * size
            example_count += size
            predictions.append(pred.detach().cpu().reshape(-1).numpy())
            targets.append(batch["ddg"].detach().cpu().reshape(-1).numpy())

        if example_count == 0:
            raise ValueError("Cannot score an empty abCAN M515 split.")

        scores = calculate_metrics(
            np.concatenate(predictions),
            np.concatenate(targets),
        )
        return {
            "loss": loss_sum / example_count,
            "rmse": float(scores["rmse"]),
            "pearson_r": float(scores["pearson_r"]),
            "spearman_rho": float(scores["spearman_rho"]),
            "n": example_count,
        }

    def train(self) -> Dict[str, float]:
        if self.test_loader is None:
            raise ValueError(
                "A held-out M515 test loader is required; it is scored only after training."
            )

        logger.info(
            "Training abCAN M515 for up to %d epochs on %s.",
            self.epochs,
            self.device,
        )
        best_val_rmse = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            val_rmse = val_metrics["rmse"]
            improved = self.early_stopping(val_rmse)

            logger.info(
                "Epoch %d/%d — train loss %.4f — val RMSE %.4f — val PCC %.4f",
                epoch,
                self.epochs,
                train_loss,
                val_rmse,
                val_metrics["pearson_r"],
            )

            if improved:
                best_val_rmse = val_rmse
                torch.save(self.model.state_dict(), self.checkpoint_path)

            if self.early_stopping.early_stop:
                logger.info("Early stopping at epoch %d.", epoch)
                break

        if not self.checkpoint_path.is_file():
            raise RuntimeError("Training ended without producing a best checkpoint.")

        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)

        # The test split is not used for early stopping or model selection.
        test_metrics = self.evaluate(self.test_loader)
        result = {
            "benchmark": "abcan_m515",
            "split": "test",
            "rmse_kcal_mol": test_metrics["rmse"],
            "pcc": test_metrics["pearson_r"],
            "spearman_rho": test_metrics["spearman_rho"],
            "n": test_metrics["n"],
            "best_validation_rmse_kcal_mol": best_val_rmse,
            "checkpoint": str(self.checkpoint_path),
        }
        with self.results_path.open("w", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2)

        logger.info(
            "Held-out abCAN M515 — RMSE %.4f kcal/mol — PCC %.4f (n=%d)",
            result["rmse_kcal_mol"],
            result["pcc"],
            result["n"],
        )
        logger.info("Saved benchmark metrics to %s.", self.results_path)
        return result
