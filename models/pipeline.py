from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import torch
import os
import wandb
from typing import Any, Optional, TYPE_CHECKING
import time

from models import (
    get_model,
    get_siamese,
    get_siamese_name,
    train_siamese,
)
from models.pl_model import Siamese_Node
from models.config import SiameseMode, OptimizationConfig
from loaders.data_generator import GAP_Generator
from loaders import siamese_loader, get_data
import loaders.data_generator as dg
from toolbox.utils import save_json, load_json
import numpy as np
from toolbox.metrics import all_qap_chain

DEFAULT_PATIENCE = 10
DEFAULT_EPS = 0.001
DEFAULT_SIZE_SEED = 20

def _negate_B_channel(dataset):
    """Negate channel 0 (adjacency matrix) of graph B for each sample in-place.

    This converts a QAP minimization problem into a GAP maximization problem,
    allowing the same loss and inference code to handle both.

    Works with both plain lists and Base_Generator/GAP_Generator objects
    (which store samples in a .data attribute).
    """
    samples = dataset.data if hasattr(dataset, "data") else dataset
    for i, sample in enumerate(samples):
        g2 = sample[1].clone()
        g2[0, :, :] = -g2[0, :, :]
        if len(sample) == 3:
            samples[i] = (sample[0], g2, sample[2])
        else:
            samples[i] = (sample[0], g2)
    return dataset


@dataclass
class LoopResult:
    best_model: Any
    best_data: Any
    best_nloop: int
    all_qap: Any
    all_ind_data: Optional[np.ndarray] = None
    all_nce_data: Optional[np.ndarray] = None
    all_faq_data: Optional[np.ndarray] = None
    all_times: Optional[np.ndarray] = None
    best_faq_perm: Optional[np.ndarray] = None


def _make_opt_cfg(cfg_training, lr_override: float | None = None) -> OptimizationConfig:
    """Build an OptimizationConfig from a Hydra training config."""
    lr = lr_override if lr_override is not None else cfg_training.lr
    return OptimizationConfig(
        lr=lr,
        scheduler_decay=cfg_training.scheduler_decay,
        scheduler_step=cfg_training.scheduler_step,
        # Halve lr_stop so the scheduler's min_lr sits below the EarlyStopping
        # threshold, ensuring the LR monitor triggers the stop first.
        lr_min=cfg_training.lr_stop / 2,
    )


class Pipeline(ABC):
    def __init__(self, path_models: str, num_models: int | None = None):

        self.path_models = path_models
        # Inter-link ranking key ("raw" | "degree_normalized"); set from
        # cfg.pipeline.rank_key at training time and from the saved config.json
        # at inference time, so a chain is always run the way it was trained.
        self.rank_key = "raw"
        # If True, the inter-link feedback uses a RANDOM node order instead of
        # the score-based ranking (the matching is still transported, only the
        # confidence ordering is destroyed). Ablation flag; same lifecycle.
        self.random_order = False
        if num_models:
            self.num_models = num_models
            self.list_models = []
        else:
            self.list_models = sorted(
                [
                    file
                    for file in os.listdir(self.path_models)
                    if file.endswith(".ckpt")
                ]
            )
            self.num_models = len(self.list_models)

        self.set_device()

    def set_device(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Device: {self.device}")

    @abstractmethod
    def train(self, cfg: DictConfig) -> None:
        pass


class Chaining(Pipeline):
    """Iterative chaining pipeline for graph alignment.

    Args:
        path_models: Directory to store/load model checkpoints.
        num_models: Number of models to train. If None, counts existing checkpoints.
        use_labels: If True, use labeled training with validation set (Siamese_Node).
            If False, use no-label Sinkhorn-based training (Siamese_Node_NL).
        negate_B: If True, negate channel 0 of graph B after loading data.
            This converts QAP minimization into GAP maximization so the same
            loss and inference code handles both problems.
    """

    def __init__(
        self,
        path_models: str,
        num_models: int | None = None,
        use_labels: bool = True,
        negate_B: bool = False,
    ):
        super().__init__(path_models, num_models)
        self.use_labels = use_labels
        self.negate_B = negate_B

    @property
    def _siamese_mode(self) -> SiameseMode:
        if self.use_labels:
            return SiameseMode.LABELED
        return SiameseMode.UNLABELED

    def build_ind(
        self,
        data,
        siamese,
        *,
        verbose: bool = False,
        compute_nce: bool = False,
        use_faq: bool = False,
        compute_faq: bool = False,
        size_seed: int = DEFAULT_SIZE_SEED,
    ) -> "tuple":
        """Run a single inference pass and return (new_data, indices, nce_scores, faq_scores).

        Returns
        -------
        new_data:
            Dataset updated with the predicted node correspondences.
        indices:
            Per-sample index assignments when ``verbose=True``, else ``None``.
        nce_scores:
            Array of NCE scores when ``compute_nce=True``, else ``None``.
        faq_scores:
            Array of FAQ scores when ``compute_faq=True``, else ``None``.
        """
        loader = siamese_loader(data, batch_size=self.batch_size, shuffle=False)
        result = dg.all_ind(
            loader,
            siamese,
            self.device,
            compute_nce=compute_nce,
            use_faq=use_faq,
            compute_faq=compute_faq,
            verbose=verbose,
            size_seed=size_seed,
            rank_key=self.rank_key,
            random_order=self.random_order,
        )
        new_data = dg.make_data_from_ind_label(data, result.indices)
        return (
            new_data,
            result.indices if verbose else None,
            result.nce_scores,
            result.faq_scores,
        )

    def train_data(self, data_train, siamese, L, data_val=None):
        train_loader = siamese_loader(
            data_train, batch_size=self.batch_size, shuffle=True
        )
        val_loader = None
        if data_val is not None:
            val_loader = siamese_loader(
                data_val, batch_size=self.batch_size, shuffle=False
            )

        train_siamese(
            train_loader,
            siamese,
            self.device,
            self.path_models,
            self.cfg.training.epochs,
            self.cfg.training.log_freq,
            L,
            self.cfg.training.lr_stop,
            self.cfg.training.wandb,
            val_loader=val_loader,
        )

        new_train, *_ = self.build_ind(data_train, siamese)
        new_val = None
        if data_val is not None:
            new_val, *_ = self.build_ind(data_val, siamese)
        if self.cfg.training.wandb:
            wandb.finish()
        if new_val is not None:
            return new_train, new_val
        return new_train

    def train(self, cfg: DictConfig, path_dataset: str) -> None:
        self.path_dataset = path_dataset
        self.cfg = cfg
        self.rank_key = str(OmegaConf.select(cfg, "pipeline.rank_key", default="raw"))
        self.random_order = bool(OmegaConf.select(cfg, "pipeline.random_order", default=False))
        self.batch_size = self.cfg.training.batch_size
        self.saving = True
        node_embedder = get_model(self.cfg.model)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        save_json(os.path.join(self.path_models, "config.json"), config_dict)

        mode = self._siamese_mode
        opt_cfg = _make_opt_cfg(self.cfg.training)

        siamese = get_siamese(node_embedder, opt_cfg=opt_cfg, mode=mode)

        data_train, data_val = get_data(
            self.cfg.dataset, self.path_dataset, self.saving, split="train_val"
        )
        if self.negate_B:
            _negate_B_channel(data_train)
            if data_val is not None:
                _negate_B_channel(data_val)
        if self.use_labels:
            new_train, new_val = self.train_data(
                data_train, siamese, L=0, data_val=data_val
            )
        else:
            # NL: use all data for training (no validation monitoring)
            new_train = self.train_data(data_train, siamese, L=0)
            new_val = None

        lr_subsequent = getattr(self.cfg.training, "lr_subsequent", self.cfg.training.lr)
        opt_cfg_sub = _make_opt_cfg(self.cfg.training, lr_override=lr_subsequent)
        siamese = get_siamese(node_embedder, opt_cfg=opt_cfg_sub, mode=mode)
        for i in range(1, self.num_models):
            if self.use_labels:
                new_train, new_val = self.train_data(
                    new_train, siamese, L=i, data_val=new_val
                )
            else:
                new_train = self.train_data(new_train, siamese, L=i)

    def _check_improvement(self, delta, current_max_nce, eps, stop, patience):
        """Check early stopping criterion with safe division."""
        if current_max_nce > eps:
            if delta / current_max_nce > eps:
                return patience
            else:
                return stop - 1
        else:
            # NCE too small for relative comparison, use absolute
            if delta > eps:
                return patience
            else:
                return stop - 1

    def _refresh_checkpoints(self):
        """Refresh checkpoint list if not yet populated (e.g. after training)."""
        if not self.list_models:
            self.list_models = sorted(
                f for f in os.listdir(self.path_models) if f.endswith(".ckpt")
            )
            self.num_models = len(self.list_models)

    def loop(
        self,
        cfg_data: DictConfig,
        path_dataset: str,
        L: int | None = None,
        N_max: int | None = None,
        patience: int = DEFAULT_PATIENCE,
        verbose: bool = False,
        eps: float = DEFAULT_EPS,
        batch_size: int | None = None,
        ind: int | None = None,
        compute_faq: bool = False,
        timing: bool = False,
        use_faq_warmstart: bool = False,
        random_order: bool | None = None,
    ) -> LoopResult:
        config = load_json(os.path.join(self.path_models, "config.json"))
        # Run the chain the way it was trained (ranking key + order); an explicit
        # random_order overrides the stored config (for inference-time ablation).
        self.rank_key = config.get("pipeline", {}).get("rank_key", "raw")
        self.random_order = (
            config.get("pipeline", {}).get("random_order", False)
            if random_order is None
            else random_order
        )
        data_test = get_data(cfg_data, path_dataset, split="test")
        if self.negate_B:
            _negate_B_channel(data_test)
        if ind is None:
            if batch_size:
                self.batch_size = batch_size
            else:
                self.batch_size = config["training"]["batch_size"]
        else:
            data_test.data = data_test.data[ind]
            self.batch_size = 1

        test_loader = siamese_loader(
            data_test, batch_size=self.batch_size, shuffle=False
        )

        self._refresh_checkpoints()

        if L is not None:
            if L > self.num_models:
                raise ValueError(
                    f"L={L} exceeds num_models={self.num_models}"
                )
            L = min(L, self.num_models)
        else:
            L = self.num_models

        if verbose or compute_faq:
            all_ind_data = []
            all_nce_data = []
            all_faq_data = []

        current_max_nce = -np.inf
        best_model = None
        best_data = data_test
        best_nloop = 0
        stop = patience
        if timing:
            start_time = time.time()
            all_times = []

        _load_mode = self._siamese_mode

        if use_faq_warmstart:
            best_faq_perm = self._faq_warmstart_loop(
                data_test, config, L, patience, eps, _load_mode,
            )
            return LoopResult(
                best_model=None,
                best_data=data_test,
                best_nloop=0,
                all_qap=None,
                best_faq_perm=best_faq_perm,
            )

        for loop_idx, model_name in enumerate(self.list_models[:L]):
            siamese = get_siamese_name(
                os.path.join(self.path_models, model_name), config["model"],
                mode=_load_mode,
            )
            new_data_test, current_ind, all_nce, all_faq = self.build_ind(
                data_test,
                siamese,
                verbose=verbose,
                compute_nce=True,
                compute_faq=compute_faq,
            )
            test_nce = all_nce.mean()
            print(f"Model {model_name} has test nce: {test_nce}")
            if compute_faq:
                print(f"Model {model_name} has test faq: {all_faq.mean()}")
                all_faq_data.append(all_faq)
            delta = test_nce - current_max_nce
            if timing:
                elapsed_time = time.time() - start_time
                all_times.append(elapsed_time)
                start_time = time.time()
                print(f"Time for model {model_name}: {elapsed_time} seconds")
            if delta > 0:
                current_max_nce = test_nce
                best_model = siamese
                best_data = data_test
                best_nloop = loop_idx
            stop = self._check_improvement(delta, current_max_nce, eps, stop, patience)
            if stop == 0:
                break
            data_test = new_data_test

            if verbose or compute_faq:
                if verbose:
                    all_ind_data.append(current_ind)
                    all_nce_data.append(all_nce)
            test_loader = siamese_loader(
                data_test, batch_size=self.batch_size, shuffle=False
            )

        if N_max and stop > 0:
            for i in range(N_max):
                new_data_test, current_ind, all_nce, all_faq = self.build_ind(
                    data_test,
                    siamese,
                    verbose=verbose,
                    compute_nce=True,
                    compute_faq=compute_faq,
                )
                test_nce = all_nce.mean()
                print(f"Model {model_name}-{i} has test nce: {test_nce}")
                if compute_faq:
                    print(f"Model {model_name}-{i} has test faq: {all_faq.mean()}")
                    all_faq_data.append(all_faq)
                delta = test_nce - current_max_nce
                if timing:
                    elapsed_time = time.time() - start_time
                    all_times.append(elapsed_time)
                    start_time = time.time()
                    print(
                        f"Time for model {model_name}-{i}: {elapsed_time} seconds"
                    )
                if delta > 0:
                    current_max_nce = test_nce
                    best_model = siamese
                    best_data = data_test
                    best_nloop += 1
                stop = self._check_improvement(
                    delta, current_max_nce, eps, stop, patience
                )
                if stop == 0:
                    break
                if i == N_max - 1:
                    break
                data_test = new_data_test
                if verbose or compute_faq:
                    if verbose:
                        all_ind_data.append(current_ind)
                        all_nce_data.append(all_nce)
                test_loader = siamese_loader(
                    data_test, batch_size=self.batch_size, shuffle=False
                )

        test_loader = siamese_loader(best_data, batch_size=1, shuffle=False)
        eval_result = all_qap_chain(test_loader, best_model, best_model.device)
        print(f"Best model has (average) nce: {eval_result.qap.mean()}")
        if len(eval_result.acc) > 0:
            print(f"Best model has acc: {eval_result.acc.mean()}")
            print(f"Best model has accmax: {eval_result.accmax}")

        return LoopResult(
            best_model=best_model,
            best_data=best_data,
            best_nloop=best_nloop,
            all_qap=eval_result.qap,
            all_ind_data=np.array(all_ind_data) if verbose else None,
            all_nce_data=np.array(all_nce_data) if verbose else None,
            all_faq_data=(
                np.array(all_faq_data) if compute_faq and all_faq_data else None
            ),
            all_times=np.array(all_times) if timing else None,
        )

    @staticmethod
    def _update_positional_encoding(data, perm):
        """Update positional encoding so matched nodes share the same rank.

        Sets graph1 node i to rank i/n and graph2 node perm[i] to rank i/n,
        encoding the current best matching into the positional channel.

        Args:
            data: Single sample tuple (tensor1, tensor2[, label]).
            perm: (n,) permutation array — graph1 node i matches graph2 node perm[i].

        Returns:
            Updated sample tuple with new positional encoding.
        """
        from loaders.representations import adjacency_matrix_to_tensor_representation_ind

        ind1 = np.arange(len(perm))  # identity: node i gets rank i/n
        ind2 = perm                   # node perm[i] gets rank i/n
        new_g1 = adjacency_matrix_to_tensor_representation_ind(data[0], ind1)
        new_g2 = adjacency_matrix_to_tensor_representation_ind(data[1], ind2)
        if len(data) == 3:
            return (new_g1, new_g2, data[2])
        return (new_g1, new_g2)

    def _faq_warmstart_loop(
        self,
        data_test,
        config: dict,
        L: int,
        patience: int,
        eps: float,
        load_mode: SiameseMode,
    ) -> Optional[np.ndarray]:
        """FAQ warm-start chaining loop.

        After each model iteration, runs FAQ refinement on the GNN scores and
        encodes the FAQ-refined permutation into the positional channel so the
        next model sees strong matching hints.

        Note: when negate_B is True, data_test contains -B in channel 0 of
        graph 2. We extract the original A and B (un-negated) for the FAQ
        solver, which needs the true matrices.

        Returns:
            Best FAQ-refined permutation, or None if no improvement found.
        """
        from toolbox.utils import perm2mat
        from scipy.optimize import linear_sum_assignment, quadratic_assignment

        device = self.device
        # Extract original A and B from the single test sample.
        # Channel 0 of graph 2 may be negated (if negate_B), so undo that.
        sample = data_test[0]
        A = sample[0][0].numpy()
        B_stored = sample[1][0].numpy()
        B = -B_stored if self.negate_B else B_stored
        n = A.shape[0]

        def qap_obj(perm):
            return (A * B[perm, :][:, perm]).sum()

        ckpts = self.list_models[:L]

        # FAQ from scratch (computed once for reference)
        res_scratch = quadratic_assignment(A, B, method="faq")
        faq_scratch_obj = qap_obj(res_scratch["col_ind"])

        best_faq_warm = np.inf
        best_perm = None
        stop = patience

        print(f"\nFAQ warm-start chaining (n={n}):")
        print(f"  {'Iter':<6} {'LAP':>14} {'FAQ warm':>14} {'FAQ scratch':>14} {'best?':>6}")

        data_iter = data_test
        for i, ckpt in enumerate(ckpts):
            siamese = get_siamese_name(
                os.path.join(self.path_models, ckpt), config["model"], mode=load_mode
            )
            siamese = siamese.to(device)
            siamese.eval()

            loader_i = siamese_loader(data_iter, batch_size=1, shuffle=False)
            with torch.no_grad():
                for batch in loader_i:
                    data1, data2 = batch[0], batch[1]
                    data1["input"] = data1["input"].to(device)
                    data2["input"] = data2["input"].to(device)
                    rawscores = siamese(data1, data2)
                    weight = torch.log_softmax(rawscores, -1)[0].cpu().numpy()

            _, col_ind_i = linear_sum_assignment(-weight)

            Pp = perm2mat(col_ind_i)
            res_warm = quadratic_assignment(A, B, method="faq", options={"P0": Pp})
            faq_perm = res_warm["col_ind"]
            faq_warm_obj = qap_obj(faq_perm)

            delta = best_faq_warm - faq_warm_obj  # positive = improvement
            is_best = ""
            if delta > 0:
                best_faq_warm = faq_warm_obj
                best_perm = faq_perm.copy()
                is_best = "*"

            if best_faq_warm < np.inf and best_faq_warm > eps:
                improved = (delta / best_faq_warm) > eps
            else:
                improved = delta > eps
            stop = patience if improved else stop - 1

            print(
                f"  {i:<6} {qap_obj(col_ind_i):>14.0f}"
                f" {faq_warm_obj:>14.0f}"
                f" {faq_scratch_obj:>14.0f}"
                f" {is_best:>6}"
            )

            if stop == 0:
                print(f"  Early stopping at iteration {i}.")
                break

            # Encode FAQ-refined permutation into positional encoding
            data_iter = [self._update_positional_encoding(data_iter[0], faq_perm)]

        return best_perm

    def loop_siamese(
        self,
        dataset: list,
        siamese: Siamese_Node,
        N_max: int | None = None,
        patience: int = DEFAULT_PATIENCE,
        verbose: bool = False,
        eps: float = DEFAULT_EPS,
        ind: int | None = None,
    ) -> LoopResult:

        if ind is None:
            data_test = dataset
        else:
            data_test = [dataset[ind]]
        self.batch_size = 1

        stop = patience
        current_max_nce = 0
        best_model = siamese
        best_data = data_test
        best_nloop = 0
        all_ind_data = []
        for i in range(N_max):
            new_data_test, current_ind, all_nce, _ = self.build_ind(
                data_test, siamese, verbose=verbose, compute_nce=True
            )
            test_nce = all_nce.mean()
            print(f"Model {i} has test nce: {test_nce}")
            delta = test_nce - current_max_nce
            if delta > 0:
                current_max_nce = test_nce
                best_model = siamese
                best_data = data_test
                best_nloop += 1
            stop = self._check_improvement(delta, current_max_nce, eps, stop, patience)
            if stop == 0:
                break
            if i == N_max - 1:
                break
            data_test = new_data_test
            if verbose:
                all_ind_data.append(current_ind)
            test_loader = siamese_loader(
                data_test, batch_size=self.batch_size, shuffle=False
            )
        del data_test

        return LoopResult(
            best_model=best_model,
            best_data=best_data,
            best_nloop=best_nloop,
            all_qap=None,
            all_ind_data=np.array(all_ind_data) if verbose else None,
        )


# Backwards-compatible alias for the no-label variant.
Chaining_NL = lambda path_models, num_models=None, negate_B=False: Chaining(
    path_models, num_models, use_labels=False, negate_B=negate_B
)


class Streaming(Pipeline):
    def __init__(self, path_models: str, num_models: int | None = None):
        super().__init__(path_models, num_models)

    def train_data(self, data_train, data_val, siamese, L):
        train_loader = siamese_loader(
            data_train, batch_size=self.batch_size, shuffle=True
        )
        val_loader = siamese_loader(
            data_val, batch_size=self.batch_size, shuffle=False
        )

        train_siamese(
            train_loader,
            siamese,
            self.device,
            self.path_models,
            self.cfg.training.epochs,
            self.cfg.training.log_freq,
            L,
            self.cfg.training.lr_stop,
            self.cfg.training.wandb,
            val_loader=val_loader,
        )

        if self.cfg.training.wandb:
            wandb.finish()

    def train(self, cfg: DictConfig, path_dataset: str) -> None:
        self.path_dataset = path_dataset
        self.cfg = cfg
        self.rank_key = str(OmegaConf.select(cfg, "pipeline.rank_key", default="raw"))
        self.random_order = bool(OmegaConf.select(cfg, "pipeline.random_order", default=False))
        self.batch_size = self.cfg.training.batch_size
        self.saving = cfg.saving
        node_embedder = get_model(self.cfg.model)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        save_json(os.path.join(self.path_models, "config.json"), config_dict)

        opt_cfg = _make_opt_cfg(self.cfg.training)
        siamese = get_siamese(node_embedder, opt_cfg=opt_cfg)

        data_train, data_val = get_data(
            self.cfg.dataset, self.path_dataset, self.saving
        )
        self.train_data(data_train, data_val, siamese, L=0)

        siamese = get_siamese(node_embedder, opt_cfg=opt_cfg)
        for i in range(1, self.num_models):
            data_train, data_val = get_data(
                self.cfg.dataset, self.path_dataset, self.saving
            )
            self.train_data(data_train, data_val, siamese, L=i)

    def test(self, cfg_data: DictConfig, path_dataset: str, verbose: bool = False):
        data_test = get_data(cfg_data, path_dataset, split="test")
        config = load_json(os.path.join(self.path_models, "config.json"))
        test_loader = siamese_loader(data_test, batch_size=1, shuffle=False)
        model_name = self.list_models[-1]
        siamese = get_siamese_name(
            os.path.join(self.path_models, model_name), config["model"]
        )
        return all_qap_chain(test_loader, siamese, self.device, verbose)
