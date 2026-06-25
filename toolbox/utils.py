import os
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def check_dir(dir_path: Union[str, Path]) -> None:
    """Ensure a directory exists, creating it and any parents if needed."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_json(json_file: Union[str, Path]) -> Dict[str, Any]:
    """Load and return the contents of a JSON file as a dictionary."""
    with open(json_file) as f:
        return json.load(f)


def save_json(json_file: Union[str, Path], data: Any) -> None:
    """Save data to a JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(data, f)


def perm2mat(p: np.ndarray) -> np.ndarray:
    """Convert permutation vector to permutation matrix.

    Args:
        p: (n,) integer array, a permutation of range(n).
    Returns:
        (n, n) float array with P[i, p[i]] = 1.
    """
    n = len(p)
    P = np.zeros((n, n))
    P[np.arange(n), p] = 1.0
    return P


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible inference.

    Also seeds NumPy's and Python's global generators, which makes any
    networkx graph draw (py_random_state) deterministic as a fallback. The
    dataset generators take an explicit Generator derived from this seed, so
    this is mainly belt-and-suspenders for the inference path.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
