"""Dataset classes for graph alignment."""

from typing import Optional
import torch
import numpy as np
import pandas as pd
import os
import tqdm
from more_itertools import chunked
import toolbox.utils as utils
from loaders.load_utils import masking_noseed, recursive_tolist
from loaders.representations import adjacency_matrix_to_tensor_representation
from loaders.generators import GENERATOR_FUNCTIONS, NOISE_FUNCTIONS


def permute(tensor, p):
    """Apply permutation p to both row and column dimensions of a 3-D tensor."""
    return tensor[:, p, :][:, :, p]


def permute_perm(label, p):
    """Apply permutation p to the row dimension of a 2-D label matrix."""
    return label[p, :]


def all_perm(
    data,
    rng: Optional[np.random.Generator] = None,
    per_sample: bool = True,
) -> list:
    """Apply random permutations to graph pairs.

    Args:
        data:       Iterable of batches; each batch is a list of
                    (graph_A, graph_B[, label]) tuples.
        rng:        NumPy Generator. If None, uses np.random.default_rng().
        per_sample: If True (default), generate an independent permutation per
                    sample.  If False, use one shared permutation for the whole
                    batch (legacy behavior).
    """
    rng = rng or np.random.default_rng()
    l_data = []
    for g_bs in data:
        mat_id = torch.eye(g_bs[0][0].shape[-1])
        g1 = torch.stack([g[0] for g in g_bs])
        g2 = torch.stack([g[1] for g in g_bs])
        label_mat = torch.stack([mat_id for g in g_bs])
        n_vertices = g1.shape[-1]
        if not per_sample:
            shared_p = rng.permutation(n_vertices)
        for i in range(g1.shape[0]):
            p = rng.permutation(n_vertices) if per_sample else shared_p
            g1perm = g1[i][:, p, :][:, :, p]
            labelperm = label_mat[i][p, :]
            l_data.append((g1perm, g2[i, :, :, :], labelperm))
    return l_data


class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples, no_seed=True, saving=False, label=False, seed: Optional[int] = None):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples
        self.no_seed = no_seed
        self.saving = saving
        self.label = label
        self.rng = np.random.default_rng(seed)

    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + ".parquet"
        path = os.path.join(self.path_dataset, filename)

        if os.path.exists(path):
            print("Reading dataset at {}".format(path))
            df = pd.read_parquet(path)
            df["graph_A"] = df["graph_A"].apply(recursive_tolist)
            df["graph_B"] = df["graph_B"].apply(recursive_tolist)
            has_perm = "permutation" in df.columns
            if has_perm:
                df["permutation"] = df["permutation"].apply(recursive_tolist)

            self.data = []
            for _, row in df.iterrows():
                graph_A = torch.tensor(row["graph_A"], dtype=torch.float32)
                graph_B = torch.tensor(row["graph_B"], dtype=torch.float32)
                if has_perm:
                    permutation = torch.tensor(row["permutation"], dtype=torch.float32)
                    self.data.append((graph_A, graph_B, permutation))
                else:
                    self.data.append((graph_A, graph_B))
        else:
            if not hasattr(self, 'compute_example'):
                raise FileNotFoundError(
                    f"Dataset not found at {path}. "
                    f"Base_Generator cannot generate data — only GAP_Generator can. "
                    f"Check that the path and data_subdir in your config are correct."
                )
            print("Creating dataset at {}".format(path))
            l_data = self.create_dataset()
            if self.saving:
                print("Saving dataset at {}".format(path))
                utils.check_dir(self.path_dataset)

                # Convert list of tuples to DataFrame
                structured_data = []
                for item_tuple in l_data:
                    row_dict = {
                        "graph_A": item_tuple[0].tolist(),
                        "graph_B": item_tuple[1].tolist(),
                        "permutation": item_tuple[2].tolist(),
                    }
                    structured_data.append(row_dict)

                df = pd.DataFrame(structured_data)
                df.to_parquet(path, index=False)

            self.data = l_data

    def remove_file(self):
        os.remove(os.path.join(self.path_dataset, self.name + ".pkl"))

    def create_dataset(self, bs=5):
        # same permutation for each batch of size bs
        l_data = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            l_data.append(example)
        return all_perm(chunked(iter(l_data), bs), rng=self.rng)

    def __getitem__(self, i):
        """Fetch sample at index i"""
        if self.no_seed:
            masking_noseed(self.data[i][0])
            masking_noseed(self.data[i][1])
        return self.data[i]

    def __len__(self):
        """Get dataset length"""
        return len(self.data)


class GAP_Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """

    def __init__(
        self, name, cfg_data, path_dataset, no_seed=True, saving=True, label=True, seed: Optional[int] = None
    ):
        self.generative_model = cfg_data.generative_model
        self.noise_model = cfg_data.noise_model
        self.edge_density = cfg_data.edge_density
        self.noise = cfg_data.noise
        num_examples = cfg_data[name].num_examples
        self.n_vertices = cfg_data.n_vertices
        subfolder_name = f"GAP_{self.generative_model}_{self.noise_model}_{num_examples}_{self.n_vertices}_{self.noise}_{self.edge_density}"
        # Seed-aware cache: distinct seeds must not collide on the same cached dataset.
        if seed is not None:
            subfolder_name += f"_seed{seed}"
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples, no_seed, saving, label, seed=seed)
        self.data = []

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        try:
            g, W, new_density = GENERATOR_FUNCTIONS[self.generative_model](
                self.edge_density, self.n_vertices, rng=self.rng
            )
        except KeyError:
            raise ValueError(
                "Generative model {} not supported".format(self.generative_model)
            )
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, new_density, rng=self.rng)
        except KeyError:
            raise ValueError("Noise model {} not supported".format(self.noise_model))
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)
