#!/usr/bin/env python3
"""Per-sample failure overlap between two methods, from a per-sample reproduction
JSONL (reproduce_results.py). For each ER-Reg cell, classify every graph pair as
solved (acc > --thr) by method A and by method B, and print the 2x2 contingency.
Because sample index i is the same pair across methods, this answers "are the
pairs A fails the same ones B fails?".

    python repro/failure_overlap.py [results.jsonl] --a chfgnn_faq --b faq_dcx [--thr 0.5]
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "repro_seed0.jsonl")


def main():
    ap = argparse.ArgumentParser(description="Per-sample failure overlap of two methods.")
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--a", default="chfgnn_faq", help="Method A (default chfgnn_faq).")
    ap.add_argument("--b", default="faq_dcx", help="Method B (default faq_dcx).")
    ap.add_argument("--thr", type=float, default=0.5, help="Solved if acc > thr (default 0.5).")
    args = ap.parse_args()
    A, B, thr = args.a, args.b, args.thr

    print(f"# solved = acc > {thr};  A = {A}, B = {B}")
    print(f"{'cell':14} {'n':>3} {'both':>5} {'A_only':>7} {'B_only':>7} {'neither':>8}")
    b_only_total = 0
    for line in open(args.path):
        r = json.loads(line)
        if r["table"] != "ER-Reg":
            continue
        m = r["methods"]
        a = [x > thr for x in m[A]["acc"]]
        b = [x > thr for x in m[B]["acc"]]
        both = sum(1 for x, y in zip(a, b) if x and y)
        a_only = sum(1 for x, y in zip(a, b) if x and not y)
        b_only = sum(1 for x, y in zip(a, b) if y and not x)
        neither = sum(1 for x, y in zip(a, b) if not x and not y)
        b_only_total += b_only
        print(f"{r['family'] + '@' + str(r['noise']):14} {len(a):>3} {both:>5} {a_only:>7} {b_only:>7} {neither:>8}")

    print()
    if b_only_total == 0:
        print(f"{B}_only == 0 in every cell  ->  {A} solves a STRICT SUPERSET of {B}'s pairs")
        print(f"(it never fails a pair {B} solves; the unsolved pairs are shared).")
    else:
        print(f"{B} solves {b_only_total} pair(s) that {A} misses (no strict dominance).")


if __name__ == "__main__":
    main()
