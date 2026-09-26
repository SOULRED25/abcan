"""
Main training entry point for abCAN-v2 SE(3) Dual-Stream architecture.
"""

import os
import torch
import logging
from torch.utils.data import DataLoader
from config import load_config
from models.abcan_v2_se3 import AbCANv2_SE3
from training.loop import Trainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing abCAN-v2 SE(3) Training Pipeline...")
    config = load_config()
    
    # Check for the extracted databases
    db_dir = config["dataset"]["database_dir"]
    if not os.path.exists(os.path.join(db_dir, "sabdab2")):
        logger.error("SAbDab2 database missing. Ensure extraction script finished.")
        return
        
    if not os.path.exists(os.path.join(db_dir, "abagym")):
        logger.error("AbAGym database missing. Ensure extraction script finished.")
        return
        
    exp_csv = config["dataset"]["experimental"]["csv_path"]
    if not os.path.exists(exp_csv):
        logger.error(f"Experimental dataset CSV missing: {exp_csv}")
        logger.error("Please place the 2,777 entry experimental CSV in the database folder.")
        return
        
    logger.info("All databases found. Initializing model...")
    model = AbCANv2_SE3(config)
    
    logger.info("Preparing datasets...")
    from data.dataset import AbCANDatasetSE3
    
    train_dataset = AbCANDatasetSE3(config, split="train")
    val_dataset = AbCANDatasetSE3(config, split="test") # using test/val depending on CSV
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    logger.info("Starting training loop...")
    # trainer.train() # Uncomment this line when ready to fully train on GPU

if __name__ == "__main__":
    main()
