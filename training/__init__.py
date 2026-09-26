"""
abCAN-v2 — Training Module
Contains custom losses and the main training loop.
"""

from training.losses import CombinedLoss
from training.loop import Trainer

__all__ = ["CombinedLoss", "Trainer"]
