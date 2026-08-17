"""Post-hoc merge of the FGWAlign baseline into a reproduction JSONL.

FGWAlign (Tang et al., "Fused Gromov-Wasserstein Alignment for Graph Edit
Distance Computation and Beyond", PVLDB 18(11), 2025) is **not vendored**: the
upstream repository (https://github.com/squareRoot3/FGWAlign) has no license,
so this driver imports it from a clone the user provides via
``--fgwalign-path``. Its core function needs only torch + pot + numpy — all
already in this project's environment.

Everything else mirrors ``repro.add_bapg`` (same seeded replay, same noise-0
edge-count fingerprint gate, same merge semantics): the new ``methods`` key is
``fgwalign``. Protocol notes:

- ``sparse=True`` always: FGWAlign's dense mode overflows float32 at n=500
  (``exp(-cost/0.01)`` on the complement-graph term → NaN → segfault inside
  POT's C EMD). Sparse mode drops the complement term, which does not change
  the optimum over permutations (both terms are affine in edge overlap).
- FGWAlign is stochastic (random-exploration restarts); we call
  ``seed_everything(record seed)`` before **every pair**, so runs are exactly
  reproducible and independent of pair order.
- The returned alignment is already a hard permutation matrix; it is decoded
  with the same orientation as the other baselines (argmax over the transposed
  plan — verified by exact recovery on planted isomorphisms). There is no unit
  test because CI has no FGWAlign clone; the noise-0 table cells play the
  isomorphism-recovery role.

Usage::

    python -m repro.add_fgwalign --fgwalign-path ~/git/FGWAlign \
        --out merged.jsonl [--light] [--family sparse]
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

from repro.add_bapg import DEFAULT_IN, _check_fingerprint, _pairs, _replay_cell
from repro.reproduce_results import SYNTHETIC, _arr, _method
from toolbox.utils import seed_everything

METHOD_KEY = "fgwalign"


def evaluate_fgwalign(FGWAlign, g1, g2, planted_perm, light=False):
    """Per-pair FGWAlign evaluation — same metric definitions as the other
    baselines (``g2[i, j] ~ g1[pl[i], pl[j]]``, acc vs planted, edge overlap)."""
    import torch

    pl = planted_perm
    n = len(pl)
    ged, trans = FGWAlign(torch.tensor(g1, dtype=torch.float32),
                          torch.tensor(g2, dtype=torch.float32),
                          sparse=True, light=light)
    T = trans.to_dense().cpu().numpy() if trans.is_sparse else trans.cpu().numpy()
    col = np.argmax(T.T, axis=1)
    return {
        "acc": np.sum(pl == col) / n,
        "nce": (g2 * g1[col, :][:, col]).sum() / 2,
        "ged": ged,
    }


def main():
    ap = argparse.ArgumentParser(description="Merge the FGWAlign baseline into a repro JSONL.")
    ap.add_argument("--fgwalign-path", required=True,
                    help="Path to a clone of https://github.com/squareRoot3/FGWAlign")
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=sorted(SYNTHETIC) + ["all"], default="all")
    ap.add_argument("--data-dir", default="./data/prepared")
    ap.add_argument("--light", action="store_true",
                    help="FGWAlign's official light variant (20 solver iterations).")
    ap.add_argument("--force", action="store_true",
                    help=f"Recompute records that already have {METHOD_KEY}.")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    src = os.path.join(os.path.expanduser(args.fgwalign_path), "src")
    if not os.path.isfile(os.path.join(src, "FGWAlign.py")):
        raise SystemExit(f"{src}/FGWAlign.py not found — clone "
                         f"https://github.com/squareRoot3/FGWAlign and pass its path")
    sys.path.insert(0, src)
    from FGWAlign import FGWAlign  # noqa: E402

    if os.path.abspath(args.out) == os.path.abspath(args.inp):
        raise SystemExit("--out must differ from --in (this script never edits its input)")
    families = sorted(SYNTHETIC) if args.family == "all" else [args.family]

    records = [json.loads(line) for line in open(args.inp)]
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for rec in records:
            selected = rec.get("table") == "ER-Reg" and rec.get("family") in families
            if selected and "methods" not in rec:
                raise SystemExit(
                    f"record {rec.get('family')}@{rec.get('noise')} has no 'methods' dict "
                    f"(old flat-schema JSONL?) — this script needs the per-sample schema")
            if selected and (args.force or METHOD_KEY not in rec["methods"]):
                print(f"{rec['family']}@{rec['noise']} ({rec['num_examples']} pairs)")
                raw = _replay_cell(rec, args.data_dir)
                if not args.skip_validation and rec["noise"] == 0:
                    _check_fingerprint(rec, raw)
                acc, nce, secs = [], [], []
                for g1, g2, pl in _pairs(raw.data):
                    seed_everything(rec["seed"])  # per-pair: order-independent runs
                    t0 = time.perf_counter()
                    r = evaluate_fgwalign(FGWAlign, g1, g2, pl, light=args.light)
                    secs.append(time.perf_counter() - t0)
                    acc.append(r["acc"])
                    nce.append(r["nce"])
                rec["methods"][METHOD_KEY] = _method(acc, nce)
                rec["methods"][METHOD_KEY]["time"] = _arr(secs, 2)  # wall s/pair
                print(f"  -> {METHOD_KEY}{' (light)' if args.light else ''}: "
                      f"mean acc {np.mean(acc):.4f}, mean nce {np.mean(nce):.1f}, "
                      f"mean {np.mean(secs):.1f} s/pair")
                del raw
                gc.collect()
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
