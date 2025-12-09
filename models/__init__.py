import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from models.utils import Network
from models.block_net import node_embedding_node_pos, block_res_mem
from models.pl_model import Siamese_Node
from models.pl_model_nl import Siamese_Node_NL

get_node_emb = {
    "node_embedding_node_pos": node_embedding_node_pos,
}

get_block_inside = {
    "block_res_mem": block_res_mem,
}


def get_model(cfg_model, original_features_num=2):
    try:
        node_emb_type = get_node_emb[cfg_model["type"]]
    except KeyError:
        raise NotImplementedError(
            f"node embedding {cfg_model['type']} is not implemented"
        )

    try:
        block_inside = get_block_inside[cfg_model["block_inside"]]
    except KeyError:
        raise NotImplementedError(
            f"block inside {cfg_model['block_inside']} is not implemented"
        )

    node_emb_args = {
        "original_features_num": original_features_num,
        "num_blocks": cfg_model["num_blocks"],
        "in_features": cfg_model["in_features"],
        "depth_of_mlp": cfg_model["depth_of_mlp"],
        "block_inside": block_inside,
    }
    node_emb_dic = {"input": (None, []), "ne": node_emb_type(**node_emb_args)}
    return Network(node_emb_dic)


def get_siamese(model):
    return Siamese_Node(model)


def get_siamese_name(path, config):
    return Siamese_Node.load_from_checkpoint(path, node_emb=get_model(config))


def get_siamese_nl(model):
    return Siamese_Node_NL(model)


def get_siamese_name_nl(path, config):
    return Siamese_Node_NL.load_from_checkpoint(path, node_emb=get_model(config))


def train_siamese(
    train_loader,
    val_loader,
    siamese,
    device,
    path_models,
    max_epochs,
    log_every_n_steps,
    L,
    lr_stop=1e-7,
    wandb=False,
):
    model_name = f"siamese_{L:02d}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_acc",
        dirpath=path_models,
        filename=model_name + "-{epoch}-{val_loss:.2f}-{val_acc:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    monitor_key = "lr-Adam"
    lr_es = EarlyStopping(
        monitor=monitor_key,
        mode="min",
        stopping_threshold=lr_stop,
        patience=100,
        check_on_train_epoch_end=True,
    )
    if wandb:
        project_name = os.path.basename(path_models.rstrip(os.sep))
        logger = WandbLogger(
            project=project_name, name=model_name, save_dir=path_models
        )
    else:
        logger = CSVLogger(path_models, name=model_name)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=max_epochs,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor, lr_es],
    )
    trainer.fit(siamese, train_loader, val_loader)


def train_siamese_nl(
    train_loader,
    siamese,
    device,
    path_models,
    max_epochs,
    log_every_n_steps,
    L,
    lr_stop=1e-7,
    wandb=False,
):
    model_name = f"siamese_{L:02d}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="train_acc",
        dirpath=path_models,
        filename=model_name + "-{epoch}-{train_loss:.2f}-{train_acc:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    monitor_key = "lr-Adam"
    lr_es = EarlyStopping(
        monitor=monitor_key,
        mode="min",
        stopping_threshold=lr_stop,
        patience=100,
        check_on_train_epoch_end=True,
    )
    if wandb:
        project_name = os.path.basename(path_models.rstrip(os.sep))
        logger = WandbLogger(
            project=project_name, name=model_name, save_dir=path_models
        )
    else:
        logger = CSVLogger(path_models, name=model_name)
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=max_epochs,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor, lr_es],
    )
    trainer.fit(siamese, train_loader)


def test_siamese(test_loader, siamese, device, path_logs):
    logger = CSVLogger(path_logs)
    trainer = pl.Trainer(accelerator=device, precision="16-mixed", logger=logger)
    trainer.test(siamese, test_loader)
    return trainer.callback_metrics["test_acc"]
