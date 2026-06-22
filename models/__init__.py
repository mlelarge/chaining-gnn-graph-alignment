from __future__ import annotations

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
from models.mp_block_net import node_embedding_mpgnn
from models.pl_model import Siamese_Node, Siamese_Node_NL
from models.config import SiameseMode, OptimizationConfig

get_node_emb = {
    "node_embedding_node_pos": node_embedding_node_pos,
    "node_embedding_mpgnn": node_embedding_mpgnn,
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

    # Build keyword arguments from the config, resolving block_inside if present.
    node_emb_args = {
        "original_features_num": original_features_num,
        "num_blocks": cfg_model["num_blocks"],
        "in_features": cfg_model["in_features"],
    }

    # FGNN-specific keys
    if "block_inside" in cfg_model:
        try:
            node_emb_args["block_inside"] = get_block_inside[cfg_model["block_inside"]]
        except KeyError:
            raise NotImplementedError(
                f"block inside {cfg_model['block_inside']} is not implemented"
            )
    if "depth_of_mlp" in cfg_model:
        node_emb_args["depth_of_mlp"] = cfg_model["depth_of_mlp"]

    # MP-GNN-specific keys
    if "conv_type" in cfg_model:
        node_emb_args["conv_type"] = cfg_model["conv_type"]
    if "num_heads" in cfg_model:
        node_emb_args["num_heads"] = cfg_model["num_heads"]

    node_emb_dic = {"input": (None, []), "ne": node_emb_type(**node_emb_args)}
    return Network(node_emb_dic)


_SIAMESE_CLASS = {
    SiameseMode.LABELED: Siamese_Node,
    SiameseMode.UNLABELED: Siamese_Node_NL,
}


def get_siamese(node_emb, opt_cfg: OptimizationConfig | None = None,
                mode: SiameseMode = SiameseMode.LABELED):
    """Factory for Siamese models.

    Args:
        node_emb: Node embedding network (output of get_model).
        opt_cfg: OptimizationConfig for learning rate schedule.
        mode: SiameseMode selecting class + loss.

    Returns:
        A Siamese model instance.
    """
    cls = _SIAMESE_CLASS[mode]
    model = cls(node_emb) if opt_cfg is None else cls(node_emb, opt_cfg)
    return model


def get_siamese_name(path, config, opt_cfg: OptimizationConfig | None = None,
                     mode: SiameseMode = SiameseMode.LABELED):
    """Load a Siamese model from a checkpoint.

    Args:
        path: Path to the ``.ckpt`` checkpoint file.
        config: Model config dict (passed to get_model).
        opt_cfg: Optional OptimizationConfig.
        mode: SiameseMode selecting class + loss.

    Returns:
        A loaded Siamese model instance.
    """
    # Allow unpickling of OptimizationConfig (PyTorch >= 2.6 defaults to weights_only=True)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OptimizationConfig])

    node_emb = get_model(config)
    cls = _SIAMESE_CLASS[mode]
    model = cls.load_from_checkpoint(path, node_emb=node_emb)
    if opt_cfg is not None:
        # frozen dataclass — replace the attribute on the Lightning module
        object.__setattr__(model, "opt_cfg", opt_cfg)
    return model


def train_siamese(
    train_loader,
    siamese,
    device,
    path_models,
    max_epochs,
    log_every_n_steps,
    L,
    lr_stop=1e-7,
    wandb=False,
    val_loader=None,
):
    model_name = f"siamese_{L:02d}"
    use_labels = not isinstance(siamese, Siamese_Node_NL)
    if val_loader is not None:
        if use_labels:
            monitor_metric = "val_acc"
            monitor_mode = "max"
            filename_metrics = "-{epoch}-{val_loss:.2f}-{val_acc:.2f}"
        else:
            monitor_metric = "val_loss"
            monitor_mode = "min"
            filename_metrics = "-{epoch}-{val_loss:.2f}"
    else:
        if use_labels:
            monitor_metric = "train_acc"
            monitor_mode = "max"
            filename_metrics = "-{epoch}-{train_loss:.2f}-{train_acc:.2f}"
        else:
            monitor_metric = "train_loss"
            monitor_mode = "min"
            filename_metrics = "-{epoch}-{train_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        mode=monitor_mode,
        monitor=monitor_metric,
        dirpath=path_models,
        filename=model_name + filename_metrics,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    monitor_key = "lr-Adam"
    lr_es = EarlyStopping(
        monitor=monitor_key,
        mode="min",
        stopping_threshold=lr_stop,
        patience=max_epochs,
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


def test_siamese(test_loader, siamese, device, path_logs):
    logger = CSVLogger(path_logs)
    trainer = pl.Trainer(accelerator=device, precision="16-mixed", logger=logger)
    trainer.test(siamese, test_loader)
    return trainer.callback_metrics["test_acc"]
