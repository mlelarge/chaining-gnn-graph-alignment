import pytorch_lightning as pl
import torch
import torch.nn as nn
from toolbox.metrics import accuracy_max
from models.loss import combined_loss_with_sinkhorn


class Siamese_Node(pl.LightningModule):
    def __init__(self, node_emb):
        super().__init__()

        self.node_embedder = node_emb

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.metric = accuracy_max

    def set_training_mode(
        self, lr=1e-3, scheduler_decay=0.5, scheduler_step=3, lr_stop=2e-5
    ):
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        # Halve lr_stop so the scheduler's min_lr sits below the EarlyStopping
        # threshold, ensuring the LR monitor triggers the stop first.
        self.lr_stop = lr_stop / 2

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.scheduler_decay,
                    patience=self.scheduler_step,
                    min_lr=self.lr_stop,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


class Siamese_Node_NL(Siamese_Node):
    """No-label variant: uses Sinkhorn-based loss and monitors train_loss only."""

    def __init__(self, node_emb):
        super().__init__(node_emb)
        self.loss_fn = combined_loss_with_sinkhorn

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
        # (acc, n) = self.metric(M, batch[2])
        # self.log("train_acc", acc / n)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.scheduler_decay,
                    patience=self.scheduler_step,
                    min_lr=self.lr_stop,
                ),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }
