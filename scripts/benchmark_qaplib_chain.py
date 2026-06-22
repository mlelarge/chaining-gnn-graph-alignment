"""
Benchmark the chaining GNN pipeline on QAPlib instances.

For each instance:
  1. Generates training data if not cached
  2. Trains a chaining NL model
  3. Runs inference and evaluates against FAQ baselines
  4. Compares against best-known QAPlib solutions

Usage:
    python scripts/benchmark_qaplib_chain.py --instances tai20a tai25a
    python scripts/benchmark_qaplib_chain.py --all --max-size 40
    python scripts/benchmark_qaplib_chain.py --instances tai40a --skip-train
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scripts.make_qaplib_data import read_dat_file, generate_instance_data, save_parquet, matrix_to_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chaining GNN on QAPlib instances"
    )
    parser.add_argument(
        "--qap-data-dir",
        default="/Users/lelarge/data/qapdata/",
        help="Path to QAPlib .dat files",
    )
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="Parquet output dir (default: <qap-data-dir>/parquet_data/)",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Root dir for experiments (default: home directory)",
    )
    parser.add_argument(
        "--instances", nargs="+", default=None, help="Instance names"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all .dat files"
    )
    parser.add_argument(
        "--max-size", type=int, default=None, help="Skip instances with n > max-size"
    )
    parser.add_argument(
        "--best-known-path",
        default=None,
        help="Path to qaplib_best_known.json",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training, only run inference"
    )
    parser.add_argument(
        "--skip-datagen", action="store_true", help="Skip data generation"
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    )
    parser.add_argument("--samples-per-noise", type=int, default=200)
    parser.add_argument("--num-models", type=int, default=2, help="Number of chaining iterations (L)")
    parser.add_argument("--output", default="benchmark_chain_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from pathlib import Path
    root_dir = Path(args.root_dir) if args.root_dir else Path.home()
    pb_dir = root_dir / "experiments-gnn-gap"
    data_pb_dir = pb_dir / "data"
    parquet_dir = args.parquet_dir or os.path.join(args.qap_data_dir, "parquet_data")

    # Load best-known solutions
    best_known = {}
    bk_path = args.best_known_path
    if bk_path is None:
        # Try default location
        bk_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "QAP", "qaplib_best_known.json",
        )
    if os.path.exists(bk_path):
        with open(bk_path) as f:
            best_known = json.load(f)
        print(f"Loaded {len(best_known)} best-known solutions")

    # Collect instances
    if args.all:
        dat_files = sorted(glob.glob(os.path.join(args.qap_data_dir, "*.dat")))
        instances = [os.path.splitext(os.path.basename(f))[0] for f in dat_files]
    elif args.instances:
        instances = args.instances
    else:
        parser.error("Specify --instances or --all")

    rng = np.random.default_rng(args.seed)
    results = []

    for instance in instances:
        dat_path = os.path.join(args.qap_data_dir, f"{instance}.dat")
        if not os.path.exists(dat_path):
            print(f"[SKIP] {instance}: .dat file not found")
            continue

        n, A, B = read_dat_file(dat_path)
        if args.max_size and n > args.max_size:
            print(f"[SKIP] {instance}: n={n} > {args.max_size}")
            continue

        print(f"\n{'='*60}")
        print(f"Instance: {instance} (n={n})")
        print(f"{'='*60}")

        t0 = time.time()

        # Step 1: Generate data if needed
        train_path = os.path.join(parquet_dir, f"{instance}_noise.parquet")
        test_path = os.path.join(parquet_dir, f"{instance}_test.parquet")

        if not args.skip_datagen and not os.path.exists(train_path):
            print("Generating training data...")
            train_data = generate_instance_data(
                A, B, args.noise_levels, args.samples_per_noise, rng
            )
            save_parquet(train_data, train_path)
            test_data = [(matrix_to_tensor(A), matrix_to_tensor(B))]
            save_parquet(test_data, test_path)

        # Step 2: Train (imports here to avoid loading torch when only generating data)
        from models.pipeline import Chaining
        from loaders import siamese_loader
        from toolbox.metrics import qaplib_evaluate
        from toolbox.utils import check_dir

        path_models = str(pb_dir / "log_models" / instance)
        check_dir(path_models)

        # Build a minimal config for this instance
        cfg = OmegaConf.create({
            "dataset": {
                "type": "nl",
                "data_subdir": "QAP",
                "no_seed": True,
                "instance": instance,
                "train": {"name": f"{instance}_noise", "num_examples": 500},
                "val": {"name": f"{instance}_test", "num_examples": 1},
            },
            "model": {
                "arch": "fgnn",
                "num_blocks": 2,
                "original_features_num": 2,
                "in_features": 64,
                "out_features": 64,
                "depth_of_mlp": 2,
            },
            "training": {
                "batch_size": 16,
                "epochs": 50,
                "lr": 1e-3,
                "lr_subsequent": 1e-4,
                "lr_stop": 1e-6,
                "scheduler_decay": 0.5,
                "scheduler_step": 5,
                "log_freq": 10,
                "wandb": False,
            },
            "pipeline": {
                "L": args.num_models,
                "path_models": f"log_models/{instance}",
            },
        })

        if not args.skip_train:
            print("Training chaining NL model...")
            chain = Chaining(path_models, cfg.pipeline.L, use_labels=False)
            chain.train(cfg, str(data_pb_dir))
        else:
            chain = Chaining(path_models, use_labels=False)

        # Step 3: Inference
        print("Running inference...")
        loop_result = chain.loop(cfg.dataset, str(data_pb_dir))

        # Step 4: Evaluate on original QAPlib matrices
        test_loader = siamese_loader(loop_result.best_data, batch_size=1, shuffle=False)
        qap_result = qaplib_evaluate(
            test_loader, loop_result.best_model, loop_result.best_model.device,
            A, B,
        )

        elapsed = time.time() - t0
        best_obj = min(
            qap_result.obj_chain, qap_result.obj_faq_warm, qap_result.obj_faq_scratch
        )

        row = {
            "instance": instance,
            "n": n,
            "obj_chain": qap_result.obj_chain,
            "obj_faq_warm": qap_result.obj_faq_warm,
            "obj_faq_scratch": qap_result.obj_faq_scratch,
            "best_obj": best_obj,
            "best_nloop": loop_result.best_nloop,
            "time_s": round(elapsed, 1),
        }

        bk_val = best_known.get(instance)
        if bk_val is not None:
            row["best_known"] = bk_val
            row["gap_chain_pct"] = round((qap_result.obj_chain - bk_val) / bk_val * 100, 2)
            row["gap_best_pct"] = round((best_obj - bk_val) / bk_val * 100, 2)

        results.append(row)

        print(f"  Chain: {qap_result.obj_chain}  FAQ-warm: {qap_result.obj_faq_warm}  "
              f"FAQ-scratch: {qap_result.obj_faq_scratch}  ({elapsed:.1f}s)")
        if bk_val is not None:
            print(f"  Best known: {bk_val}  Gap: {row['gap_best_pct']}%")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print(f"Processed {len(results)} instances")


if __name__ == "__main__":
    main()
