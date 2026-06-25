"""Reproduce the paper's results tables with a fixed seed (Phase 6 / reproducibility).

Runs the full experiment grid deterministically and writes one JSON record per
(table-cell) to ``--out``. Designed to be run on a CPU cluster for the heavy
cells (dense Erdős–Rényi in particular); the JSON is then turned into the README
tables. Every cell is seeded (``--seed``), so the numbers are reproducible (they
will be *close to* but not identical to the paper, which used no seed).

Per **synthetic** cell (one trained model per family, swept over noise) it reports
acc/nce for the six rows of the ER/Regular table:
  Proj(D_cx), FAQ(D_cx)          — convex-relaxation baselines (evaluate_faq_inits)
  FGNN Proj,  FGNN FAQ           — the single (first) network        (all_qap_chain)
  ChFGNN Proj, ChFGNN FAQ        — the chained model                 (chaining loop)

Per **real-world** cell it reports FAQ(D_cx), Max-nce (baselines), the
dataset-specific ChFGNN and the transferred ChFGNN-ER4. FUGAL/SGWL stay external.

Usage::

    python -m repro.reproduce_results --all --seed 0 --out repro_results.json
    python -m repro.reproduce_results --family sparse --seed 0 --out sparse.json
    python -m repro.reproduce_results --real --seed 0 --out real.json
"""

import argparse
import gc
import json
import os

import numpy as np

from loaders import get_data, siamese_loader
from models import get_siamese_name
from models.config import SiameseMode
from models.pipeline import Chaining
from run_inference import download_release
from toolbox.baselines import evaluate_faq_inits
from toolbox.metrics import all_qap_chain
from toolbox.utils import seed_everything

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Synthetic families: release tag + noise sweep (matches tab:ER-Reg).
SYNTHETIC = {
    "sparse":  {"release": "v1.0.0-er500-d4-pn0.22",  "noises": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]},
    "dense":   {"release": "v1.0.0-er500-d80-pn0.24", "noises": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]},
    "regular": {"release": "v1.0.0-reg500-d10-pn0.11", "noises": [0, 0.05, 0.1, 0.15, 0.2]},
}

# Real-world cells (tab:realworld-noisy): the dataset-specific ChFGNN release, the
# prepare_data dataset key + test parquet, and the paper noise label.
REAL = [
    {"cell": "ca-netscience@0.1",  "release": "v1.1.0-canetscience-pn0.1", "subdir": "ca-netscience", "test": "ca_nets_noise1_test", "prep": "ca-netscience"},
    {"cell": "ca-netscience@0.2",  "release": "v1.1.0-canetscience-pn0.2", "subdir": "ca-netscience", "test": "ca_nets_noise2_test", "prep": "ca-netscience"},
    {"cell": "inf-euroroad@0.1",   "release": "v1.1.0-euroroad-pn0.1",     "subdir": "inf-euroroad",  "test": "road_noise1_test",  "prep": "inf-euroroad"},
    {"cell": "inf-euroroad@0.2",   "release": "v1.1.0-euroroad-pn0.2",     "subdir": "inf-euroroad",  "test": "road_noise2_test",  "prep": "inf-euroroad"},
    {"cell": "yeast25LC@0.05",     "release": "v1.1.0-yeast25lc-pn0.05",   "subdir": "MultiMagna",    "test": "yeast0_25_noise005_test", "prep": "multimagna-noisy"},
    {"cell": "yeast25LC@0.1",      "release": "v1.1.0-yeast25lc-pn0.1",    "subdir": "MultiMagna",    "test": "yeast0_25_noise01_test",  "prep": "multimagna-noisy"},
]
ER4_RELEASE = "v1.0.0-er500-d4-pn0.22"  # transferred ChFGNN-ER4


def _emit(out_path, record):
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("  ->", json.dumps(record))


def _baselines(data, maxiter_faq=30):
    """FAQ baselines over a dataset's (A, B, planted) pairs (mirrors run_baseline)."""
    keys = ["acc_proj", "nce_proj", "acc_dcx", "nce_dcx", "nce_max"]
    acc = {k: [] for k in keys}
    for item in data:
        g1 = item[0][0].cpu().numpy()
        g2 = item[1][0].cpu().numpy()
        pl = np.argmax(item[2].cpu().numpy(), 0)
        r = evaluate_faq_inits(g1, g2, pl, maxiter_faq=maxiter_faq)
        for k in keys:
            acc[k].append(r[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _model_metrics(loader, model):
    device = next(model.parameters()).device
    ev = all_qap_chain(loader, model, device)
    return {
        "acc_faq": float(np.mean(ev.acc)), "nce_faq": float(np.mean(ev.qap)),
        "acc_proj": float(np.mean(ev.accd)), "nce_proj": float(np.mean(ev.d)),
    }


def run_synthetic(family, args, out_path):
    spec = SYNTHETIC[family]
    path_models = download_release(spec["release"], args.checkpoint_dir)
    config = json.load(open(os.path.join(path_models, "config.json")))
    from omegaconf import OmegaConf

    for noise in (args.noises or spec["noises"]):
        seed_everything(args.seed)
        ds = config["dataset"]
        cfg = OmegaConf.create({
            "type": "synthetic", "n_vertices": ds["n_vertices"],
            "generative_model": ds["generative_model"], "noise_model": ds["noise_model"],
            "edge_density": ds["edge_density"], "noise": noise, "seed": args.seed,
            "test": {"num_examples": args.num_examples},
        })
        data_dir = os.path.join(args.data_dir, f"{family}_seed{args.seed}")
        raw = get_data(cfg, data_dir, saving=True, split="test")

        base = _baselines(raw.data)                                   # Proj(D_cx), FAQ(D_cx)
        chain = Chaining(path_models)
        m0 = get_siamese_name(os.path.join(path_models, chain.list_models[0]),
                              config["model"], mode=SiameseMode.LABELED).to(chain.device)
        fgnn = _model_metrics(siamese_loader(raw.data, batch_size=1, shuffle=False), m0)
        loop_kw = {"N_max": args.N_max} if args.N_max is not None else {}
        res = chain.loop(cfg, data_dir, **loop_kw)                    # ChFGNN (chained)
        chf = _model_metrics(siamese_loader(res.best_data, batch_size=1, shuffle=False),
                             res.best_model)
        _emit(out_path, {
            "table": "ER-Reg", "family": family, "noise": noise,
            "num_examples": args.num_examples, "seed": args.seed,
            "proj_dcx": [round(base["acc_proj"], 4), round(base["nce_proj"], 1)],
            "faq_dcx":  [round(base["acc_dcx"], 4), round(base["nce_dcx"], 1)],
            "fgnn_proj": [round(fgnn["acc_proj"], 4), round(fgnn["nce_proj"], 1)],
            "fgnn_faq":  [round(fgnn["acc_faq"], 4), round(fgnn["nce_faq"], 1)],
            "chfgnn_proj": [round(chf["acc_proj"], 4), round(chf["nce_proj"], 1)],
            "chfgnn_faq":  [round(chf["acc_faq"], 4), round(chf["nce_faq"], 1)],
        })
        # Free the per-cell data/models before the next noise level (avoids OOM).
        del raw, chain, m0, res, fgnn, chf, base
        gc.collect()


def run_real(args, out_path):
    import subprocess
    import sys
    from omegaconf import OmegaConf

    for c in REAL:
        seed_everything(args.seed)
        # Build the (seeded) test parquet for this dataset if absent.
        subprocess.run([sys.executable, "-m", "repro.prepare_data", "--dataset", c["prep"],
                        "--seed", str(args.seed), "--output-dir", args.data_dir], check=True)
        cfg = OmegaConf.create({"type": "real", "data_subdir": c["subdir"], "no_seed": True,
                                "test": {"name": c["test"], "num_examples": args.num_examples}})
        raw = get_data(cfg, args.data_dir, saving=False, split="test")
        base = _baselines(raw.data)                                   # FAQ(D_cx), Max
        rec = {"table": "realworld-noisy", "cell": c["cell"], "seed": args.seed,
               "num_examples": len(raw.data),
               "faq_dcx": [round(base["acc_dcx"], 4), round(base["nce_dcx"], 1)],
               "max_nce": round(base["nce_max"], 1)}
        for label, rel in [("chfgnn", c["release"]), ("chfgnn_er4", ER4_RELEASE)]:
            pm = download_release(rel, args.checkpoint_dir)
            cfgm = json.loads(json.dumps(OmegaConf.to_container(cfg)))
            loop_kw = {"N_max": args.N_max} if args.N_max is not None else {}
            res = Chaining(pm).loop(OmegaConf.create(cfgm), args.data_dir, **loop_kw)
            ev = all_qap_chain(siamese_loader(res.best_data, batch_size=1, shuffle=False),
                               res.best_model, res.best_model.device)
            rec[label] = [round(float(np.mean(ev.acc)), 4), round(float(np.mean(ev.qap)), 1)]
        _emit(out_path, rec)


def _cuda():
    import torch
    return torch.cuda.is_available()


def main():
    ap = argparse.ArgumentParser(description="Reproduce paper tables with a fixed seed.")
    ap.add_argument("--all", action="store_true", help="All synthetic families + real-world.")
    ap.add_argument("--family", choices=sorted(SYNTHETIC), help="One synthetic family.")
    ap.add_argument("--real", action="store_true", help="Real-world cells only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-examples", type=int, default=100, help="Test pairs per cell (default 100).")
    ap.add_argument("--noises", type=float, nargs="+", default=None, help="Override the synthetic noise sweep (quick checks).")
    ap.add_argument("--N-max", type=int, default=80, help="Chaining refinement cap (default 80, like run_inference; the loop early-stops). N_max=None would SKIP refinement.")
    ap.add_argument("--checkpoint-dir", default="./checkpoints")
    ap.add_argument("--data-dir", default="./data/prepared")
    ap.add_argument("--out", default="repro_results.jsonl", help="Output JSONL file (appended).")
    args = ap.parse_args()

    families = sorted(SYNTHETIC) if (args.all or (not args.family and not args.real)) else ([args.family] if args.family else [])
    print(f"Writing results to {args.out} (seed={args.seed}, num_examples={args.num_examples})")
    open(args.out, "w").close()  # start fresh — don't append to a previous run's file
    for fam in families:
        print(f"=== synthetic: {fam} ===")
        run_synthetic(fam, args, args.out)
    if args.all or args.real:
        print("=== real-world ===")
        run_real(args, args.out)


if __name__ == "__main__":
    main()
