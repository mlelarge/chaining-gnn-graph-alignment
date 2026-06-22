"""Self-contained inference on real-world graph pairs (Phase 4).

Mirror of ``run_inference.py`` for the real datasets. Either downloads a
pretrained checkpoint from a GitHub release (``--release``) or uses a local
checkpoint directory (``--checkpoint-dir``), loads a prepared test parquet
(built by ``repro/prepare_data.py``), runs the chaining loop, and reports the
number of common edges (nce) and node accuracy — the metrics in the paper's
real-world tables.

The released checkpoints carry a ``config.json`` whose dataset block names the
``data_subdir`` and test parquet, so ``--release`` is self-describing; with a
local ``--checkpoint-dir`` you pass ``--data-subdir`` / ``--test-name`` yourself.

Usage::

    python -m repro.prepare_data --dataset ca-netscience              # once
    python run_inference_real.py --release v1.1.0-canetscience-pn0.1  # no GPU needed

    # or against a local checkpoint dir:
    python run_inference_real.py \
        --checkpoint-dir ~/experiments-gnn-gap/cleps_files/label_canet_noise1 \
        --data-subdir ca-netscience --test-name ca_nets_noise1_test
"""

import argparse
import json
import os

import numpy as np
from omegaconf import OmegaConf

from loaders import siamese_loader
from models.pipeline import Chaining
from toolbox.metrics import all_qap_chain


def main():
    ap = argparse.ArgumentParser(description="Real-world chaining inference.")
    ap.add_argument("--release", default=None, help="GitHub release tag to download (alternative to --checkpoint-dir).")
    ap.add_argument("--checkpoint-dir", default=None, help="Local path_models dir (siamese_*.ckpt + config.json).")
    ap.add_argument("--checkpoint-cache", default="./checkpoints", help="Where --release downloads to.")
    ap.add_argument("--data-subdir", default=None, help="Dataset subdir (default: from release config.json).")
    ap.add_argument("--test-name", default=None, help="Test parquet stem (default: from release config.json).")
    ap.add_argument("--data-dir", default="./data", help="Base dir holding <data-subdir>/<test-name>.parquet.")
    ap.add_argument("--num-examples", type=int, default=20, help="Number of test pairs to evaluate.")
    ap.add_argument("--L", type=int, default=None, help="Max chaining models (default: all).")
    ap.add_argument("--N-max", type=int, default=None, help="Max refinement iterations with the best model.")
    ap.add_argument("--negate-B", action="store_true", help="Negate channel 0 of graph B (default: off, the real-data convention).")
    args = ap.parse_args()

    if args.release:
        from run_inference import download_release

        path_models = download_release(args.release, args.checkpoint_cache)
        ds = json.load(open(os.path.join(path_models, "config.json"))).get("dataset", {})
        data_subdir = args.data_subdir or ds.get("data_subdir")
        test_name = args.test_name or (ds.get("test") or {}).get("name")
        if not data_subdir or not test_name:
            ap.error("release config.json lacks dataset.data_subdir / dataset.test.name; pass --data-subdir / --test-name")
    else:
        if not (args.checkpoint_dir and args.data_subdir and args.test_name):
            ap.error("provide --release, or --checkpoint-dir together with --data-subdir and --test-name")
        path_models = args.checkpoint_dir
        data_subdir = args.data_subdir
        test_name = args.test_name

    cfg_data = OmegaConf.create(
        {
            "type": "real",
            "data_subdir": data_subdir,
            "no_seed": True,
            "test": {"name": test_name, "num_examples": args.num_examples},
        }
    )

    chain = Chaining(path_models, negate_B=args.negate_B)
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
    print(f"Models:          {args.release or path_models}")
    print(f"Dataset:         {data_subdir} / {test_name}")
    print(f"Test examples:   {args.num_examples}   negate_B={args.negate_B}")
    print(f"Best chain iter: {result.best_nloop}")
    print(f"Accuracy (acc):  {acc:.4f}")
    print(f"Common edges (nce): {nce:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
