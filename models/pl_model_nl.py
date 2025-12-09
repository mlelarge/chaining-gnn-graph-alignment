import pytorch_lightning as pl
import torch
import torch.nn as nn
from toolbox.metrics import accuracy_max
from models.loss import combined_loss_with_sinkhorn


class Siamese_Node_NL(pl.LightningModule):
    def __init__(self, node_emb):
        """ """
        super().__init__()

        self.node_embedder = node_emb

        # self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.loss = combined_loss_with_sinkhorn
        self.metric = accuracy_max

    def set_training_mode(
        self, lr=1e-3, scheduler_decay=0.5, scheduler_step=3, lr_stop=2e-5
    ):
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.lr_stop = lr_stop / 2  # for optimizer to enable earlystopping

    def forward(self, x1, x2, verbose=False):
        """
        Data should be given with the shape (b,2,f,n,n)
        """
        x1 = self.node_embedder(x1)["ne/suffix"]
        x2 = self.node_embedder(x2)["ne/suffix"]
        # raw_scores = torch.einsum('bfi,bfj-> bij', x1, x2)
        raw_scores = torch.matmul(torch.transpose(x1, 1, 2), x2)
        if verbose:
            return raw_scores, x1, x2
        else:
            return raw_scores

    def training_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        # loss = self.loss(raw_scores, batch[2])
        loss, matching_loss, bisto_loss, M = self.loss(
            raw_scores,
            batch[0]["input"][:, 0, :, :],
            batch[1]["input"][:, 0, :, :],
            return_M=True,
        )
        self.log("train_loss", loss)
        self.log("train_matching_loss", matching_loss)
        self.log("train_bisto_loss", bisto_loss)
        # (acc,n) = self.metric(raw_scores, batch[2])
        (acc, n) = self.metric(M, batch[2])
        self.log("train_acc", acc / n)
        return loss

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
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
