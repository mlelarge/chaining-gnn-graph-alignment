"""Self-contained inference on real-world graph pairs (Phase 4).

Mirror of ``run_inference.py`` for the real datasets. Loads a pretrained
checkpoint directory (``siamese_*.ckpt`` + ``config.json``) and a prepared
test parquet (built by ``repro/prepare_data.py``), runs the chaining loop, and
reports the number of common edges (nce) and node accuracy — the metrics in the
paper's real-world tables.

Note: the real-data checkpoints' ``config.json`` carries only the *model* block
(the dataset block is unpopulated), so the dataset is specified on the CLI
(``--data-subdir`` / ``--test-name``) rather than read from ``config.json``.

Usage::

    python -m repro.prepare_data --dataset ca-netscience          # once
    python run_inference_real.py \
        --checkpoint-dir ~/experiments-gnn-gap/cleps_files/label_canet_noise1 \
        --data-subdir ca-netscience --test-name ca_nets_noise1_test \
        --data-dir ~/experiments-gnn-gap/data
"""

import argparse

import numpy as np
from omegaconf import OmegaConf

from loaders import siamese_loader
from models.pipeline import Chaining
from toolbox.metrics import all_qap_chain


def main():
    ap = argparse.ArgumentParser(description="Real-world chaining inference.")
    ap.add_argument("--checkpoint-dir", required=True, help="path_models dir (siamese_*.ckpt + config.json).")
    ap.add_argument("--data-subdir", required=True, help="Dataset subdir, e.g. ca-netscience / inf-euroroad / MultiMagna.")
    ap.add_argument("--test-name", required=True, help="Test parquet stem (without .parquet), e.g. ca_nets_noise1_test.")
    ap.add_argument("--data-dir", default="./data", help="Base dir holding <data-subdir>/<test-name>.parquet.")
    ap.add_argument("--num-examples", type=int, default=20, help="Number of test pairs to evaluate.")
    ap.add_argument("--L", type=int, default=None, help="Max chaining models (default: all).")
    ap.add_argument("--N-max", type=int, default=None, help="Max refinement iterations with the best model.")
    ap.add_argument("--negate-B", action="store_true", help="Negate channel 0 of graph B (match training convention).")
    args = ap.parse_args()

    cfg_data = OmegaConf.create(
        {
            "type": "real",
            "data_subdir": args.data_subdir,
            "no_seed": True,
            "test": {"name": args.test_name, "num_examples": args.num_examples},
        }
    )

    chain = Chaining(args.checkpoint_dir, negate_B=args.negate_B)
    kwargs = {}
    if args.L is not None:
        kwargs["L"] = args.L
    if args.N_max is not None:
        kwargs["N_max"] = args.N_max
    result = chain.loop(cfg_data, args.data_dir, **kwargs)

    nce = float(np.mean(result.all_qap)) if result.all_qap is not None else float("nan")
    loader = siamese_loader(result.best_data, batch_size=1, shuffle=False)
    ev = all_qap_chain(loader, result.best_model, result.best_model.device)
    acc = float(np.mean(ev.acc)) if ev.acc is not None and ev.acc.size else float("nan")

    print("\n" + "=" * 60)
    print("REAL-WORLD INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Checkpoints:     {args.checkpoint_dir}")
    print(f"Dataset:         {args.data_subdir} / {args.test_name}")
    print(f"Test examples:   {args.num_examples}   negate_B={args.negate_B}")
    print(f"Best chain iter: {result.best_nloop}")
    print(f"Accuracy (acc):  {acc:.4f}")
    print(f"Common edges (nce): {nce:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
