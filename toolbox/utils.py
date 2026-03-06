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
    """Convert a permutation vector to a permutation matrix.

    Args:
        p: 1D array where p[i] gives the column index for row i.

    Returns:
        Binary (n, n) matrix P where P[i, p[i]] = 1.
    """
    n = np.max(p.shape)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, p[i]] = 1
    return P
