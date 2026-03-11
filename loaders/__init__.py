"""Public API for the loaders package."""

import os
from typing import Optional
from torch.utils.data import DataLoader
import torch

from loaders.datasets import GAP_Generator, Base_Generator


def collate_fn(samples_list):
    has_target = len(samples_list[0]) == 3
    input1_list = [sample[0] for sample in samples_list]
    input2_list = [sample[1] for sample in samples_list]
    if has_target:
        target_list = [sample[2] for sample in samples_list]
        return (
            {"input": torch.stack(input1_list)},
            {"input": torch.stack(input2_list)},
            torch.stack(target_list),
        )
    return (
        {"input": torch.stack(input1_list)},
        {"input": torch.stack(input2_list)},
    )


def siamese_loader(
    data: list,
    batch_size: int,
    train: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    device=None,
    # legacy alias kept for backward compatibility
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Args:
        num_workers: If None, auto-detects based on os.cpu_count(). Pass 0 to disable multiprocessing.
        pin_memory:  If True (default) and device is CUDA, enables async memory transfers.
        device:      torch.device or None. If None, checks for CUDA availability.
        shuffle:     Deprecated alias for ``train``. If provided, overrides ``train``.
    """
    assert len(data) > 0
    # honour legacy shuffle kwarg
    if shuffle is not None:
        train = shuffle
    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 1))
    if device is None:
        use_pin_memory = pin_memory and torch.cuda.is_available()
    else:
        use_pin_memory = pin_memory and (hasattr(device, 'type') and device.type == "cuda")
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
    )


# ---- Unified data loading ----


def get_data(cfg_data, path_dataset, saving=True, split="train_val"):
    """
    Unified data loading entry point.

    Dispatches based on cfg_data.type:
      - "synthetic" (default): uses GAP_Generator with generative_model/noise_model
      - "real": uses Base_Generator with cfg_data.data_subdir
      - "nl": same as "real", returns train only

    Args:
        cfg_data: Hydra config with dataset parameters
        path_dataset: base data directory
        saving: whether to cache to parquet
        split: "train_val" returns (train, val), "test" returns test only,
               "train_only" returns train only
    """
    dataset_type = getattr(cfg_data, "type", "synthetic")

    if dataset_type == "synthetic":
        return _get_synthetic(cfg_data, path_dataset, saving, split)
    elif dataset_type in ("real", "nl"):
        return _get_real(cfg_data, path_dataset, saving, split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _get_synthetic(cfg_data, path_dataset, saving, split):
    if split == "train_val":
        gene_train = GAP_Generator("train", cfg_data, path_dataset, saving=saving)
        gene_train.load_dataset()
        gene_val = GAP_Generator("val", cfg_data, path_dataset, saving=saving)
        gene_val.load_dataset()
        return gene_train, gene_val
    elif split == "test":
        label = getattr(cfg_data, "label", True)
        gene_test = GAP_Generator(
            "test", cfg_data, path_dataset, saving=saving, label=label
        )
        gene_test.load_dataset()
        return gene_test
    elif split == "train_only":
        gene_train = GAP_Generator("train", cfg_data, path_dataset, saving=saving)
        gene_train.load_dataset()
        return gene_train


def _get_real(cfg_data, path_dataset, saving, split):
    # Use data_path if absolute, otherwise join with path_dataset + data_subdir
    data_path = getattr(cfg_data, "data_path", None)
    if data_path and os.path.isabs(data_path):
        real_path = data_path
    else:
        real_path = os.path.join(path_dataset, cfg_data.data_subdir)
    no_seed = getattr(cfg_data, "no_seed", True)

    if split == "train_val":
        gene_train = Base_Generator(
            name=cfg_data.train.name,
            path_dataset=real_path,
            num_examples=cfg_data.train.num_examples,
            no_seed=no_seed,
            saving=saving,
        )
        gene_train.load_dataset()
        gene_val = Base_Generator(
            name=cfg_data.val.name,
            path_dataset=real_path,
            num_examples=cfg_data.val.num_examples,
            no_seed=no_seed,
            saving=saving,
        )
        gene_val.load_dataset()
        return gene_train, gene_val
    elif split == "test":
        # Fall back to val config if test section is missing
        test_cfg = cfg_data.test if hasattr(cfg_data, "test") else cfg_data.val
        gene_test = Base_Generator(
            name=test_cfg.name,
            path_dataset=real_path,
            num_examples=test_cfg.num_examples,
            no_seed=no_seed,
            saving=saving,
        )
        gene_test.load_dataset()
        return gene_test


# ---- Backward-compatible aliases ----


def get_data_test(cfg_data, path_dataset, saving=True):
    """Deprecated: use get_data(cfg_data, path_dataset, split='test')"""
    return get_data(cfg_data, path_dataset, saving=saving, split="test")


