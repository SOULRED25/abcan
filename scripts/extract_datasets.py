"""
Utility script to extract zipped and tarred datasets for abCAN-v2.
Handles AbAGym (.zip) and SAbDab2 (.tar.gz).
"""

import os
import zipfile
import tarfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_archive(file_path: str, extract_to: str):
    """Detects archive type and extracts it to the target directory."""
    path = Path(file_path)
    target_dir = Path(extract_to)
    
    if not path.exists():
        logger.error(f"Archive not found: {path}")
        return
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if path.name.endswith('.zip'):
        logger.info(f"Extracting ZIP: {path} -> {target_dir}")
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
    elif path.name.endswith(('.tar.gz', '.tgz', '.tar.tz')):
        logger.info(f"Extracting TAR: {path} -> {target_dir}")
        with tarfile.open(path, 'r:gz') as tar_ref:
            tar_ref.extractall(target_dir)
            
    else:
        logger.error(f"Unsupported archive format: {path.name}")
        return
        
    logger.info(f"Successfully extracted to {target_dir}")

if __name__ == "__main__":
    db_dir = Path("database")
    db_dir.mkdir(exist_ok=True)
    
    # Locate AbAGym zip
    abagym_zips = list(db_dir.glob("abagym*.zip")) + list(db_dir.glob("AbAGym*.zip"))
    if abagym_zips:
        extract_archive(abagym_zips[0], "database/abagym")
    else:
        logger.warning("No AbAGym zip file found in 'database/'. Expected e.g., 'abagym.zip'")
        
    # Locate SAbDab2 tar.gz
    sabdab_tars = list(db_dir.glob("sabdab*.tar.*")) + list(db_dir.glob("SAbDab*.tar.*")) + list(db_dir.glob("splits.tar.gz"))
    if sabdab_tars:
        extract_archive(sabdab_tars[0], "database/sabdab2")
    else:
        logger.warning("No SAbDab2 tar file found in 'database/'. Expected e.g., 'sabdab2.tar.gz'")
        
    logger.info("Extraction process finished.")
