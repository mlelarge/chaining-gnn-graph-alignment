import pytorch_lightning as pl
import torch
import torch.nn as nn
from toolbox.metrics import accuracy_max

class Siamese_Node(pl.LightningModule):
    def __init__(self, node_emb):
        """
        
        """
        super().__init__()
        
        self.node_embedder = node_emb
        
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.metric = accuracy_max 

    def set_training_mode(self, lr=1e-3, scheduler_decay=0.5, scheduler_step=3, lr_stop = 2e-5):
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.lr_stop = lr_stop/2 # for optimizer to enable earlystopping
        
    def forward(self, x1, x2 , verbose =False):
        """
        Data should be given with the shape (b,2,f,n,n)
        """
        x1 = self.node_embedder(x1)['ne/suffix']
        x2 = self.node_embedder(x2)['ne/suffix']
        #raw_scores = torch.einsum('bfi,bfj-> bij', x1, x2)
        raw_scores = torch.matmul(torch.transpose(x1,1,2),x2)
        if verbose:
            return raw_scores, x1, x2
        else:
            return raw_scores
    
    def training_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log('train_loss', loss)
        (acc,n) = self.metric(raw_scores, batch[2])
        self.log("train_acc", acc/n)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log('val_loss', loss)
        (acc,n) = self.metric(raw_scores, batch[2])
        self.log("val_acc", acc/n)

    def test_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        loss = self.loss(raw_scores, batch[2])
        self.log('test_loss', loss)
        (acc,n) = self.metric(raw_scores, batch[2])
        self.log("test_acc", acc/n)
        return acc/n
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, min_lr=self.lr_stop),
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }
