"""Post-hoc merge of the BAPG-GW baseline into a reproduction JSONL.

Adds a ``bapg_proj`` entry (BAPG Gromov-Wasserstein, Li et al. ICLR 2023, via
POT — see ``toolbox/bapg.py``) to the ``methods`` dict of the selected synthetic
family's records, evaluated on the *identical seeded, ordered* test pairs as
the committed run. It replays each cell's data exactly as
``repro.reproduce_results.run_synthetic`` does — per-cell
``seed_everything(seed)`` immediately followed by the cfg build and
``get_data`` (the EdgeSwap noise of the regular family consumes Python's
*global* RNG, so this call order is load-bearing) — and needs no checkpoints
and no network: dataset parameters are fixed per family below, and the
validation gates would catch any drift.

One validation gate runs per family, at the noise-0 cell (solver-free, no
baseline is re-run): the committed ``faq_dcx`` nce at noise 0 equals each
pair's raw edge count (a solver that is right up to automorphisms matches
every edge), so the replayed edge counts must reproduce the committed values
exactly. Verified 30/30 (sparse) and 10/10 (dense) against the committed run;
uninformative on regular (D_cx collapse), where replay identity rests on
copying reproduce_results' per-cell seeding call order verbatim.

The output is written to ``--out + '.tmp'`` and renamed only on success, and
``--out`` must differ from ``--in`` (this script never edits its input).
Records outside the selected families (real-world cells, other families) pass
through unchanged, so families can be merged incrementally::

    python -m repro.add_bapg --family sparse --out merged.jsonl
    python -m repro.add_bapg --in merged.jsonl --family dense --out merged2.jsonl
"""

import argparse
import gc
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf

from loaders import get_data
from repro.reproduce_results import SYNTHETIC, _arr, _method
from toolbox.bapg import evaluate_bapg
from toolbox.utils import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "results", "repro_seed0.jsonl")

# Dataset parameters per family (n=500 for all), matching the release
# config.json of the tags in reproduce_results.SYNTHETIC and the corresponding
# conf/dataset yamls. The validation gates verify the replayed data anyway.
FAMILY_DATASETS = {
    "sparse":  {"generative_model": "ErdosRenyi", "noise_model": "ErdosRenyi", "edge_density": 0.008},
    "dense":   {"generative_model": "ErdosRenyi", "noise_model": "ErdosRenyi", "edge_density": 0.16},
    "regular": {"generative_model": "Regular",    "noise_model": "EdgeSwap",   "edge_density": 0.02},
}

METHOD_KEY = "bapg_proj"


def _replay_cell(rec, data_dir):
    """Regenerate one cell's test data exactly as reproduce_results does."""
    ds = FAMILY_DATASETS[rec["family"]]
    seed_everything(rec["seed"])
    cfg = OmegaConf.create({
        "type": "synthetic", "n_vertices": 500,
        "generative_model": ds["generative_model"], "noise_model": ds["noise_model"],
        "edge_density": ds["edge_density"], "noise": rec["noise"], "seed": rec["seed"],
        "test": {"num_examples": rec["num_examples"]},
    })
    fam_dir = os.path.join(data_dir, f"{rec['family']}_seed{rec['seed']}")
    raw = get_data(cfg, fam_dir, saving=True, split="test")
    if len(raw.data) != rec["num_examples"]:
        raise SystemExit(f"replay of {rec['family']}@{rec['noise']}: got "
                         f"{len(raw.data)} pairs, record says {rec['num_examples']}")
    return raw


def _pairs(data):
    for item in data:
        g1 = item[0][0].cpu().numpy()
        g2 = item[1][0].cpu().numpy()
        pl = np.argmax(item[2].cpu().numpy(), 0)
        yield g1, g2, pl


def _check_fingerprint(rec, raw):
    """Noise-0 gate: replayed per-pair edge counts vs committed faq_dcx nce.

    At noise 0 a solver that is right up to automorphisms matches every edge,
    so the committed nce equals the pair's raw edge count — a solver-free data
    fingerprint. The gate aborts below a half-matching threshold (chance
    matching on unrelated data is ~0); the observed match on the committed run
    is 30/30 (sparse) and 10/10 (dense). On the regular family the D_cx
    collapse makes it uninformative (committed nce is the overlap of a
    near-random permutation, not the edge count); replay identity there rests
    on copying reproduce_results' per-cell seeding call order verbatim
    (EdgeSwap consumes the global RNG).
    """
    committed = rec["methods"]["faq_dcx"]
    edges = [g1.sum() / 2 for g1, _, _ in _pairs(raw.data)]
    match = sum(1 for e, n in zip(edges, committed["nce"]) if e == n)
    n = rec["num_examples"]
    if match < n / 2:
        if rec["family"] == "regular":
            print(f"  [gate] noise-0 fingerprint uninformative on regular "
                  f"({match}/{n} — D_cx collapse); relying on seeded replay")
            return
        raise SystemExit(
            f"noise-0 fingerprint FAILED for {rec['family']}: only {match}/{n} replayed "
            f"edge counts match the committed faq_dcx nce. The regenerated data "
            f"differs from the committed run — do not merge.")
    print(f"  [gate] noise-0 fingerprint OK ({match}/{n} edge counts match)")


def main():
    ap = argparse.ArgumentParser(description="Merge the BAPG-GW baseline into a repro JSONL.")
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=sorted(SYNTHETIC) + ["all"], default="all")
    ap.add_argument("--data-dir", default="./data/prepared")
    ap.add_argument("--force", action="store_true",
                    help=f"Recompute records that already have {METHOD_KEY}.")
    ap.add_argument("--skip-validation", action="store_true",
                    help="Skip the replay gates (only if already validated on this machine).")
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.inp):
        raise SystemExit("--out must differ from --in (this script never edits its input)")
    families = sorted(SYNTHETIC) if args.family == "all" else [args.family]

    if families == ["regular"]:
        print("note: the regular family has no informative data fingerprint (D_cx "
              "collapse at noise 0); replay identity relies on the seeded call order.")

    records = [json.loads(line) for line in open(args.inp)]
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for rec in records:
            selected = rec.get("table") == "ER-Reg" and rec.get("family") in families
            if selected and "methods" not in rec:
                raise SystemExit(
                    f"record {rec.get('family')}@{rec.get('noise')} has no 'methods' dict "
                    f"(old flat-schema JSONL?) — add_bapg needs the per-sample schema")
            if selected and (args.force or METHOD_KEY not in rec["methods"]):
                print(f"{rec['family']}@{rec['noise']} ({rec['num_examples']} pairs)")
                raw = _replay_cell(rec, args.data_dir)
                if not args.skip_validation and rec["noise"] == 0:
                    _check_fingerprint(rec, raw)
                acc, nce, secs, fallbacks = [], [], [], 0
                for g1, g2, pl in _pairs(raw.data):
                    t0 = time.perf_counter()
                    r = evaluate_bapg(g1, g2, pl)
                    secs.append(time.perf_counter() - t0)
                    acc.append(r["acc_bapg"])
                    nce.append(r["nce_bapg"])
                    fallbacks += int(r.get("bapg_failed", 0))
                rec["methods"][METHOD_KEY] = _method(acc, nce)
                rec["methods"][METHOD_KEY]["time"] = _arr(secs, 2)  # wall s/pair
                note = f"  [WARNING: {fallbacks} identity-fallback pairs]" if fallbacks else ""
                print(f"  -> {METHOD_KEY}: mean acc {np.mean(acc):.4f}, "
                      f"mean nce {np.mean(nce):.1f}, mean {np.mean(secs):.1f} s/pair{note}")
                del raw
                gc.collect()
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
