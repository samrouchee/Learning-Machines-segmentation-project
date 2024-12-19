# training/config.py

from dataclasses import dataclass
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer
from typing import Callable, Optional

@dataclass
class TrainingConfig:
    model: torch.nn.Module
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    epochs: int
    optimizer: Optimizer
    criterion: Callable
    device: torch.device
    scheduler: ReduceLROnPlateau
    save_path: str = "/content/drive/MyDrive/saved_models"  # Save models to Google Drive
    model_name: str = "model.pth"  # Name of the saved model
    gradient_clipping: bool = True
    max_norm: float = 1.0
