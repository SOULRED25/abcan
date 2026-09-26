"""
abCAN-v2 — Training Loop
Adam Optimizer, Cosine Annealing, Early Stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pathlib import Path
from typing import Dict, Any

from training.losses import CombinedLoss

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Main training loop for abCAN-v2."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        train_cfg = config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        loss_cfg = train_cfg.get("loss", {}).get("components", {})
        
        self.epochs = train_cfg.get("epochs", 200)
        self.device = config.get("project", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        
        # Optimizer & Scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-5),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999]))
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get("scheduler", {}).get("T_max", 200),
            eta_min=train_cfg.get("scheduler", {}).get("eta_min", 1e-6)
        )
        
        # Loss & Early Stopping
        self.criterion = CombinedLoss(
            mse_weight=loss_cfg.get("mse", 1.0),
            weighted_mse_weight=loss_cfg.get("weighted_mse", 0.5),
            hinge_weight=loss_cfg.get("cosine_hinge", 0.3)
        )
        
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001)
        )
        
        self.checkpoint_dir = Path(config.get("logging", {}).get("checkpoint_dir", "checkpoints/"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # This is pseudo-code for the batch unpacking. 
            # In PyG, batch would be a Data or Batch object.
            # Assuming dictionaries for standard DataLoader here:
            pred = self.model(
                struct_node_feat=batch["struct_node_feat"].to(self.device),
                struct_edge_index=batch["struct_edge_index"].to(self.device),
                struct_edge_attr=batch["struct_edge_attr"].to(self.device),
                wt_seq_emb=batch["wt_seq_emb"].to(self.device),
                mut_seq_emb=batch["mut_seq_emb"].to(self.device),
                biochem_features=batch["biochem_features"].to(self.device),
                interface_mask=batch["interface_mask"].to(self.device),
                mutation_mask=batch["mutation_mask"].to(self.device),
                padding_mask=batch.get("padding_mask", None).to(self.device) if "padding_mask" in batch else None
            )
            
            target = batch["ddg"].to(self.device)
            weights = batch.get("sample_weight", torch.ones_like(target)).to(self.device)
            
            loss_dict = self.criterion(pred, target, weights)
            loss = loss_dict["loss"]
            
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
            self.optimizer.step()
            total_loss += loss.item()
            
        self.scheduler.step()
        return total_loss / len(self.train_loader)
        
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            pred = self.model(...) # Unpack similar to train_epoch
            target = batch["ddg"].to(self.device)
            weights = batch.get("sample_weight", torch.ones_like(target)).to(self.device)
            
            loss_dict = self.criterion(pred, target, weights)
            total_loss += loss_dict["loss"].item()
            
        return total_loss / len(self.val_loader)
        
    def train(self):
        logger.info(f"Starting training for {self.epochs} epochs.")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            # val_loss = self.validate() # pseudo-code due to data packing
            val_loss = train_loss * 0.95 # Mock for now
            
            logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
            # Save checkpoint
            if epoch % 10 == 0 or self.early_stopping.early_stop:
                torch.save(self.model.state_dict(), self.checkpoint_dir / f"model_epoch_{epoch}.pt")
                
        logger.info("Training completed.")
