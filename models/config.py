from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from enum import Enum, Flag, auto
from torch.utils.data import DataLoader
import numpy as np


class TrainMode(Enum):
    LABELED   = "labeled"    # CrossEntropyLoss, monitors val_loss
    UNLABELED = "unlabeled"  # Sinkhorn loss, monitors train_loss


class SiameseMode(Enum):
    LABELED   = "labeled"
    UNLABELED = "unlabeled"


class CollectionFlags(Flag):
    NONE        = 0
    NCE_SCORES  = auto()
    FAQ_SCORES  = auto()
    TIMING      = auto()
    INDICES     = auto()
    ALL = NCE_SCORES | FAQ_SCORES | TIMING | INDICES


@dataclass(frozen=True)
class OptimizationConfig:
    lr: float = 1e-3
    scheduler_decay: float = 0.5
    scheduler_step: int = 3
    lr_min: float = 1e-7

    def __post_init__(self):
        if not (0 < self.lr <= 1):
            raise ValueError(f"lr must be in (0, 1], got {self.lr}")
        if not (0 < self.lr_min < self.lr):
            raise ValueError(f"lr_min must be in (0, lr), got {self.lr_min}")


@dataclass
class TrainingConfig:
    train_loader: DataLoader
    siamese: object  # Siamese_Node | Siamese_Node_NL
    device: str
    path_models: Path
    iteration_idx: int
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    max_epochs: int = 100
    log_every_n_steps: int = 10
    val_loader: Optional[DataLoader] = None
    use_wandb: bool = False
    mode: TrainMode = TrainMode.LABELED


@dataclass
class LoopConfig:
    cfg_data: object  # DictConfig
    path_dataset: str
    path_models: str
    collect: CollectionFlags = CollectionFlags.NCE_SCORES


@dataclass
class LoopResult:
    iteration_nce: Optional[np.ndarray]
    iteration_faq: Optional[np.ndarray]
    iteration_times: Optional[list]
    iteration_indices: Optional[list]
