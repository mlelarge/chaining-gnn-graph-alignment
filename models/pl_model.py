from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from toolbox.metrics import accuracy_max
from models.loss import combined_loss_with_sinkhorn
from models.config import OptimizationConfig, SiameseMode


class Siamese_Node(pl.LightningModule):
    def __init__(self, node_emb, opt_cfg: OptimizationConfig = OptimizationConfig()):
        super().__init__()

        self.node_embedder = node_emb
        self.opt_cfg = opt_cfg
        self.mode = SiameseMode.LABELED

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.metric = accuracy_max

        # Save opt_cfg fields as flat hyperparameters.
        # Exclude node_emb (not serialisable) and opt_cfg (frozen dataclass
        # breaks Lightning's apply_to_collection during hparam logging).
        self.save_hyperparameters(ignore=["node_emb", "opt_cfg"])
        self.hparams.update(
            {
                "lr": opt_cfg.lr,
                "scheduler_decay": opt_cfg.scheduler_decay,
                "scheduler_step": opt_cfg.scheduler_step,
                "lr_min": opt_cfg.lr_min,
            }
        )

    def forward(self, x1, x2):
        """
        Data should be given with the shape (b, 2, f, n, n).
        Returns raw_scores of shape (b, n, n) - node matching score matrix.
        """
        x1 = self.node_embedder(x1)["ne/suffix"]
        x2 = self.node_embedder(x2)["ne/suffix"]
        # x1, x2: (b, f_out, n), transpose -> (b, n, f_out)
        # raw_scores: (b, n, f_out) @ (b, f_out, n) = (b, n, n)
        raw_scores = torch.matmul(torch.transpose(x1, 1, 2), x2)
        return raw_scores

    def forward_with_features(self, x1, x2):
        """Returns (raw_scores, x1_emb, x2_emb) for analysis/debugging."""
        x1 = self.node_embedder(x1)["ne/suffix"]
        x2 = self.node_embedder(x2)["ne/suffix"]
        raw_scores = torch.matmul(torch.transpose(x1, 1, 2), x2)
        return raw_scores, x1, x2

    def training_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log("train_loss", loss)
        (acc, n) = self.metric(raw_scores, batch[2])
        self.log("train_acc", acc / n)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log("val_loss", loss)
        (acc, n) = self.metric(raw_scores, batch[2])
        self.log("val_acc", acc / n)

    def test_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log("test_loss", loss)
        (acc, n) = self.metric(raw_scores, batch[2])
        self.log("test_acc", acc / n)

    def configure_optimizers(self):
        opt_cfg = self.opt_cfg
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_cfg.lr, amsgrad=False)

        monitor = "train_loss" if self.mode.is_unlabeled else "val_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=opt_cfg.scheduler_decay,
                    patience=opt_cfg.scheduler_step,
                    min_lr=opt_cfg.lr_min,
                ),
                "monitor": monitor,
                "frequency": 1,
            },
        }


class Siamese_Node_NL(Siamese_Node):
    """No-label variant: uses Sinkhorn-based loss and monitors train_loss only."""

    def __init__(self, node_emb, opt_cfg: OptimizationConfig = OptimizationConfig()):
        super().__init__(node_emb, opt_cfg)
        self.loss_fn = combined_loss_with_sinkhorn
        self.mode = SiameseMode.UNLABELED

    def training_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss, matching_loss, bisto_loss, M = self.loss_fn(
            raw_scores,
            batch[0]["input"][:, 0, :, :],
            batch[1]["input"][:, 0, :, :],
            return_M=True,
        )
        self.log("train_loss", loss)
        self.log("train_matching_loss", matching_loss)
        self.log("train_bisto_loss", bisto_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss, matching_loss, bisto_loss, M = self.loss_fn(
            raw_scores,
            batch[0]["input"][:, 0, :, :],
            batch[1]["input"][:, 0, :, :],
            return_M=True,
        )
        self.log("val_loss", loss)
        self.log("val_matching_loss", matching_loss)
        self.log("val_bisto_loss", bisto_loss)

    def test_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss, matching_loss, bisto_loss, M = self.loss_fn(
            raw_scores,
            batch[0]["input"][:, 0, :, :],
            batch[1]["input"][:, 0, :, :],
            return_M=True,
        )
        self.log("test_loss", loss)
        self.log("test_matching_loss", matching_loss)
        self.log("test_bisto_loss", bisto_loss)


