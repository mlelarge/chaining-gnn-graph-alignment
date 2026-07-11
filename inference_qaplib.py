"""
Inference script for QAPlib instances (no-label case).

Loads a trained NL chaining model, runs inference on the QAPlib test pair,
and evaluates against FAQ baselines and best-known solutions.

Usage:
    python inference_qaplib.py
    python inference_qaplib.py dataset=qaplib dataset.instance=chr25a
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from loaders import siamese_loader
from models.pipeline import Chaining
from scripts.make_qaplib_data import read_dat_file
from toolbox.metrics import qaplib_evaluate


@hydra.main(version_base=None, config_path="conf", config_name="config_nl")
def main(cfg: DictConfig):
    if cfg.root_dir is None:
        ROOT_DIR = Path.home()
    else:
        ROOT_DIR = os.path.abspath(cfg.root_dir)
    PB_DIR = os.path.join(ROOT_DIR, "experiments-gnn-gap/")
    DATA_PB_DIR = os.path.join(PB_DIR, "data/")
    path_models = os.path.join(PB_DIR, cfg.pipeline.path_models)
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Determine instance name from config
    instance = getattr(cfg.dataset, "instance", cfg.dataset.val.name.replace("_test", ""))

    # Load original QAPlib matrices (place raw .dat files under <repo>/data/qapdata/,
    # or override with `qap_data_dir=/path/to/qapdata` on the CLI).
    qap_data_dir = getattr(cfg, "qap_data_dir", os.path.join(REPO_ROOT, "data", "qapdata"))
    dat_path = os.path.join(qap_data_dir, f"{instance}.dat")
    n, A_raw, B_raw = read_dat_file(dat_path)
    print(f"Instance: {instance} (n={n})")

    # Load best-known solutions if available
    best_known = None
    best_known_path = os.path.join(
        REPO_ROOT, "extras", "qaplib", "qaplib_best_known.json"
    )
    if os.path.exists(best_known_path):
        with open(best_known_path) as f:
            bk = json.load(f)
        best_known = bk.get(instance)

    # Run chaining loop
    chain = Chaining(path_models, use_labels=False)
    result = chain.loop(cfg.dataset, DATA_PB_DIR)

    # Evaluate on original QAPlib matrices
    test_loader = siamese_loader(result.best_data, batch_size=1, shuffle=False)
    qap_result = qaplib_evaluate(
        test_loader, result.best_model, result.best_model.device, A_raw, B_raw,
    )

    # Report results
    print(f"\n{'='*50}")
    print(f"Instance: {instance} (n={n})")
    print(f"Best chaining iteration: {result.best_nloop}")
    print(f"{'='*50}")
    print(f"  Chain (LAP on GNN scores): {qap_result.obj_chain}")
    print(f"  FAQ warm (GNN init):       {qap_result.obj_faq_warm}")
    print(f"  FAQ scratch:               {qap_result.obj_faq_scratch}")
    best_obj = min(qap_result.obj_chain, qap_result.obj_faq_warm, qap_result.obj_faq_scratch)
    print(f"  Best:                      {best_obj}")

    if best_known is not None:
        gap_chain = (qap_result.obj_chain - best_known) / best_known * 100
        gap_warm = (qap_result.obj_faq_warm - best_known) / best_known * 100
        gap_scratch = (qap_result.obj_faq_scratch - best_known) / best_known * 100
        gap_best = (best_obj - best_known) / best_known * 100
        print(f"  Best known:                {best_known}")
        print(f"  Gap chain:                 {gap_chain:.2f}%")
        print(f"  Gap FAQ warm:              {gap_warm:.2f}%")
        print(f"  Gap FAQ scratch:           {gap_scratch:.2f}%")
        print(f"  Gap best:                  {gap_best:.2f}%")


if __name__ == "__main__":
    main()
