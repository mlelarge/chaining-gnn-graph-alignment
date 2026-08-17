# Plan — BAPG Gromov-Wasserstein baseline (Li et al., ICLR 2023) on sparse / dense / regular

**Goal.** Add BAPG-GW — *Li, Tang, Kong, Liu, Li, So, Blanchet, "A Convergent Single-Loop
Algorithm for Relaxation of Gromov-Wasserstein in Graph Data", ICLR 2023* (arXiv:2303.06595),
as implemented by `ot.gromov.BAPG_gromov_wasserstein` in POT — as an **in-repo baseline**,
evaluated **per-sample on the exact same seeded test pairs** as the chained GNN, on the three
published synthetic regimes (sparse ER d≈4, dense ER d≈80, regular d=10/EdgeSwap), feeding the
`repro_seed0.jsonl` → tables → per-sample-plot pipeline.

Branch: create `feature/bapg-baseline` off `main` (this is orthogonal to
`feature/single-conditioned-gnn`).

---

## 0. Verified background facts (from code recon, 2026-07-23)

These are the constraints the plan is built on; all verified against the repo.

- **Baseline template**: `toolbox/baselines.py:evaluate_faq_inits(g1, g2, planted_perm, maxiter_faq=30)`
  is the canonical per-pair evaluator. Contract: `g1`, `g2` are (n,n) numpy 0/1 symmetric
  adjacencies (channel 0 of the stored tensors), `pl = np.argmax(target, axis=0)`;
  acc `= np.sum(pl == col)/n`; nce `= (g2 * g1[col,:][:,col]).sum()/2`. Orientation is locked by
  `tests/test_baselines.py`: `evaluate_*(g1, g2, pl)` assumes `g2[i,j] ≈ g1[pl[i], pl[j]]`.
- **Soft-plan decoding convention**: the FW solution is decoded `linear_sum_assignment(-P.T)`
  (`toolbox/frank_wolfe.py:156`) — transpose first, then LAP. For a BAPG plan
  `T = BAPG_gromov_wasserstein(C1=g1, C2=g2, ...)` (rows = g1 nodes), the matching decode is
  `_, col = linear_sum_assignment(-T.T)`.
- **The canonical results artifact** is `repro/results/repro_seed0.jsonl` (one JSON record per
  table cell; `methods.<key>.{acc,nce}` are **full per-pair arrays**, index i = same graph pair
  for every method). Written by `repro/reproduce_results.py`; grids hard-coded in its
  `SYNTHETIC` dict: sparse/dense noise ∈ {0, .05, …, .35}, regular ∈ {0, .05, …, .2};
  committed run = seed 0, **30 pairs/cell (sparse, regular), 10 (dense)**, n=500.
- **Replay is exact**: no generation/eval code changed since the committed run's commit
  (`84b4911`); ER cells are seeded through an explicit `np.random.default_rng`; the regular
  family's EdgeSwap noise uses Python's **global** `random`, so it replays **iff** the per-cell
  call order of `run_synthetic` is copied verbatim: `seed_everything(0)` → build cfg (with
  `seed: 0`) → `get_data(cfg, data_dir, saving=True, split="test")`, nothing RNG-consuming in
  between. Cache path: `./data/prepared/{family}_seed0/GAP_{gen}_{noise_model}_{num}_{n}_{noise}_{density}_seed2/test.parquet`
  (noise `0` renders as int `0`). **Neither `./data/prepared/` nor `./checkpoints/` exists
  locally** — data must be regenerated (deterministic), and a BAPG-only run needs **no
  checkpoints and no network**.
- **All-or-nothing per table**: `repro/format_tables.py` and `repro/failure_overlap.py` index
  every registered method in **every** record of a table — a `bapg_*` key must be present in all
  21 ER-Reg cells or `make tables` crashes. `repro/samples_to_csv.py` and
  `failure_overlap.py --a/--b` pick up new keys with zero code change.
- **`repro/reproduce_results.py:main()` truncates `--out` at start** — never point a partial
  run at the committed jsonl.
- **`run_baseline.py` has a seed gap** (its rebuilt config has no `seed` field → silently
  regenerates unseeded data if the cache was seeded). The repro path is the one to extend.
- **POT**: `pot` is not currently a dependency. BAPG solvers exist since POT 0.9.2; numpy-2.x
  support since 0.9.4 (repo pins numpy 2.3.3). API:
  `BAPG_gromov_wasserstein(C1, C2, p=None, q=None, loss_fun="square_loss", epsilon=0.1,
  symmetric=None, G0=None, max_iter=1000, tol=1e-9, marginal_loss=False, log=False)`.
  `epsilon` is the paper's Bregman **step-size ρ**, *not* entropic regularization: too small →
  `exp` overflow → NaN plan (POT warns to increase it); larger → tighter marginals, slower.
  Returned `T` **violates the marginal constraints by design**; the returned `gw_dist` is not a
  true GW cost (can be negative) — use `T` only. Deterministic, no RNG. POT's backend accepts
  numpy or torch tensors (free GPU path if ever needed; unnecessary at n=500).
- **Paper protocol** (graph alignment experiments): raw 0/1 adjacencies, uniform marginals,
  ρ = 0.1 (no tuning), stop at relative change ≤ 1e-6, cap 2000 iters, correspondence by
  row-wise argmax of the plan. The paper tested Gaussian-partition/BA graphs — **not** ER or
  regular — so a small ε sanity sweep on our regimes is warranted before locking defaults.

---

## 1. Decisions (recommended defaults — confirm before Phase D)

| # | Decision | Recommendation |
|---|---|---|
| D1 | In-repo vs external-cited (RELEASE_PLAN.md locked decision #3 says only FAQ is vendored; FUGAL/SGWL are cited) | **In-repo** — per-sample pairing is the whole point and is impossible with quoted numbers. Add an addendum line to RELEASE_PLAN.md noting the extension of decision #3. |
| D2 | Dependency lane | `pot>=0.9.4` as a **core** dependency (`uv add "pot>=0.9.4"`), precedent: scipy's FAQ is core. Commit `pyproject.toml` + `uv.lock` together (CI runs `uv sync`). |
| D3 | Correspondence extraction rows | Two table rows mirroring the D_cx pair: **`bapg_proj`** = LAP on `-T.T` (repo convention, bijective) and **`bapg_faq`** = FAQ refined from the BAPG assignment. Compute the paper-faithful row-argmax as a diagnostic column in the pilot only (not a table row) to check how much LAP adds. |
| D4 | Rerun vs merge | **Post-hoc merge** into a copy of `repro_seed0.jsonl` (proven safe: exact data replay, no code drift; needs no checkpoints, no GPU, no network). Full `make reproduce` rerun is the fallback if we ever want to refresh every method anyway. |
| D5 | Hyperparameters | Module defaults `epsilon=0.1, max_iter=2000, tol=1e-6, loss_fun="square_loss"`, uniform marginals, default `G0`, `symmetric=None` — the paper's alignment protocol — confirmed/adjusted by the Phase C pilot. Follow the `frank_wolfe.py` "canonical settings in the docstring, exposed as kwargs, no CLI flags" pattern. |
| D6 | Real-world cells | **Defer** (optional Phase G). Synthetic first; the real cells replay deterministically too if wanted. |
| D7 | Dense n=10 statistical power | Keep n=10 for the paired table (it's what every other method has); flag in README caveats. A larger all-methods run is out of scope. |

---

## 2. Phase A — dependency + solver module

1. `uv add "pot>=0.9.4"` → commit `pyproject.toml` + `uv.lock`. Immediately confirm the
   installed POT's `BAPG_gromov_wasserstein` signature matches §0 (`python -c "import ot,
   inspect; print(ot.__version__, inspect.signature(ot.gromov.BAPG_gromov_wasserstein))"`) —
   the signature above comes from the docs, not a live install.
2. New module **`toolbox/bapg.py`** (first-class solver module, independently testable, like
   `toolbox/frank_wolfe.py`):

   ```python
   from dataclasses import dataclass
   import numpy as np
   from scipy.optimize import linear_sum_assignment
   import ot

   @dataclass
   class BAPGResult:
       T: np.ndarray          # (n, n) transport plan (may violate marginals — by design)
       col_ind: np.ndarray    # (n,) permutation, axis-0 convention (comparable to pl)
       n_iter: int | None

   def solve_bapg(A, B, epsilon=0.1, max_iter=2000, tol=1e-6, loss_fun="square_loss"):
       """Canonical settings: paper's graph-alignment protocol (rho=0.1, 2000 iters,
       relative tol 1e-6, square loss, uniform marginals, raw adjacency inputs)."""
       T, log = ot.gromov.BAPG_gromov_wasserstein(
           A, B, loss_fun=loss_fun, epsilon=epsilon,
           max_iter=max_iter, tol=tol, log=True)
       if not np.all(np.isfinite(T)):          # epsilon too small -> exp overflow
           raise FloatingPointError("BAPG produced non-finite plan; increase epsilon")
       _, col = linear_sum_assignment(-T.T)    # frank_wolfe.py:156 convention
       return BAPGResult(T, col, len(log.get("err", [])) or None)

   def evaluate_bapg(g1, g2, planted_perm, epsilon=0.1, max_iter=2000, tol=1e-6,
                     maxiter_faq=30):
       """Per-pair evaluator mirroring evaluate_faq_inits. Returns
       {"acc_bapg", "nce_bapg", "acc_bapg_faq", "nce_bapg_faq"}."""
   ```

   `evaluate_bapg` internals copy `evaluate_faq_inits` exactly: `overlap(col) =
   (g2 * g1[col,:][:,col]).sum()/2`, `acc = np.sum(pl == col)/len(pl)`; the FAQ-refined variant
   is `quadratic_assignment(g2, -g1, method="faq", options={"P0": perm2mat(col),
   "maxiter": maxiter_faq})["col_ind"]` (`perm2mat` from `toolbox/utils.py`; a permutation
   matrix is a valid doubly-stochastic `P0`, same pattern as the Max-nce row).

   **NaN policy for grid runs**: catch `FloatingPointError` in `evaluate_bapg`, retry once with
   `epsilon * 10`; if still non-finite, return the identity permutation's scores and emit a
   warning — a grid run must never crash, and the record stays complete (all-or-nothing rule).

## 3. Phase B — unit tests (`tests/test_bapg.py`)

Mirror `tests/test_baselines.py` (plain functions, standalone-run shim, `np.random.default_rng`,
no byte-exact cross-platform assertions):

1. **Isomorphism recovery** — `B = A[np.ix_(perm, perm)]` on an ER(60, 0.2) graph; assert
   `acc_bapg == 1.0` and `nce_bapg ==` edge count. *This test mechanically locks the T-vs-T.T
   orientation* — write it first.
2. **Noisy sanity** — perturbed pair: acc ∈ [0,1], nce ≤ nce_planted-style bound, results finite.
3. **Regular-graph smoke** — small random-regular pair: runs to completion, returns finite
   scores (degeneracy is allowed, crashing is not).
4. **Marginal violation tolerated** — folded into test 2 as a bijectivity assertion on
   the decoded permutation (a standalone `T.sum(axis=1) != p` assertion would be flaky:
   nothing forces the converged plan to be infeasible on easy pairs).

Run: `uv run pytest tests/ -q` (CI does the same on Linux).

## 4. Phase C — pilot: ε sensitivity + regular-family degeneracy check

Small scratch script (not committed, or committed under `notebooks/`): generate ~5 seeded pairs
per family at one mid-grid noise (sparse@0.15, dense@0.15, regular@0.1) via the same
`get_data` path, then:

- sweep `epsilon ∈ {0.05, 0.1, 0.5, 1.0}` × `max_iter ∈ {1000, 2000}`; record acc (argmax vs
  LAP vs FAQ-refined), wall-clock, and whether `log['err']` converged;
- confirm `epsilon=0.1` is a sane operating point for ER (the paper never tested ER); adjust D5
  only if the sweep clearly says so;
- **check the expected regular-family pathology**: uniform degrees + flat `G0 = pqᵀ` make the
  first square-loss gradient step uniform (same mechanism as the documented D_cx collapse,
  README caveats block). Record whether BAPG is near-random there — that's a result to report,
  not a bug — and note it for the README caveat text (Phase F).

Log the pilot outcome in this file's execution log (`EXPERIMENT_JOURNAL.md` is the
confidential conditioned-GNN branch journal — out of scope for this public branch).

## 5. Phase D — evaluation run: post-hoc merge script (`repro/add_bapg.py`)

A standalone script; **does not call `reproduce_results.main()`** (which truncates its output).

```
python -m repro.add_bapg \
    --in  repro/results/repro_seed0.jsonl \
    --out repro/results/repro_seed0_bapg.jsonl \
    --data-dir ./data/prepared
```

Per **synthetic** record (family, noise, num_examples, seed) of the input jsonl:

1. **Replay the cell** by copying `run_synthetic`'s per-cell sequence *verbatim*
   (`repro/reproduce_results.py:105-146`): `seed_everything(seed)` → build the same cfg dict
   (with `"seed": seed`) → `get_data(cfg, f"{data_dir}/{family}_seed{seed}", saving=True,
   split="test")`. Nothing RNG-consuming in between (EdgeSwap constraint). First run
   regenerates and caches all parquets (~no checkpoints or network needed).
2. **Validation gate (mandatory)**: recompute `proj_dcx`/`faq_dcx` on the replayed data for the
   first cell of each family and assert exact equality (after `_arr` rounding: acc 4 dp,
   nce 1 dp) with the committed record's arrays. This proves pair-for-pair replay identity —
   the only real risk is the regular family's global-RNG EdgeSwap, and this gate catches it.
   Abort without writing if it fails.
3. Evaluate BAPG on `raw.data` **in file order**, exactly like `_baselines()`:
   `g1 = item[0][0].cpu().numpy()`, `g2 = item[1][0].cpu().numpy()`,
   `pl = np.argmax(item[2].cpu().numpy(), 0)` → `evaluate_bapg(g1, g2, pl)`.
4. Append `"bapg_proj": {"acc": [...], "nce": [...]}` and `"bapg_faq": {...}` to the record's
   `methods` (via the same `_method`/`_arr` rounding helpers — import them), write the merged
   record to `--out`. Real-world records pass through unchanged.
5. After eyeballing `--out`: replace `repro/results/repro_seed0.jsonl` with it (git tracks the
   diff — only additions inside `methods`), keep the original in git history.

**Cost estimate**: 470 pairs total (8×30 sparse + 8×10 dense + 5×30 regular), n=500; one BAPG
iteration is two dense 500×500 matmuls, ×2000 iters ≈ a few seconds/pair on CPU → **~30–60 min
single-threaded**, plus FAQ refinement (seconds/pair). Runs on the laptop; no cluster needed.

## 6. Phase E — regenerate derived artifacts

- `make samples` → `samples.csv` picks up `bapg_*` automatically.
- `make tables CI=--ci` after adding to `repro/format_tables.py:ERREG_ROWS`:
  `("Proj(BAPG)", "bapg_proj")`, `("FAQ(BAPG)", "bapg_faq")` (rows must exist in all 21 ER-Reg
  records — they will, by construction).
- `make overlap` extra runs: `--a chfgnn_faq --b bapg_faq` (and `--a faq_dcx --b bapg_faq`).
- `make plot`: add `bapg_faq` to `M_COLOR`/`M_LABEL` and the panel-(a) method list in
  `repro/plot_samples.py`; decide whether panel (b) gains a BAPG-vs-chain paired scatter or
  stays as-is (recommend: add a third panel rather than repurposing (b)).

## 7. Phase F — documentation

- **README**: paste refreshed tables; add BAPG rows to the baselines description (cite Li et
  al. ICLR 2023 + POT ≥0.9.4 + exact settings ε=0.1, 2000 iters, tol 1e-6, square loss, uniform
  marginals, LAP/FAQ extraction); extend the caveats block with the regular-family outcome from
  the pilot; update the per-sample-analysis prose if the figure changed.
- **RELEASE_PLAN.md**: addendum to decision #3 (BAPG-GW vendored in-repo via POT, unlike
  FUGAL/SGWL, because per-sample pairing requires it). Note: the file is untracked and
  gitignored (local working doc), so the addendum stays local by design.

## 8. Phase G — optional follow-ups (separate decisions)

- **Real-world cells**: same merge pattern over the 6 real records (`prepare_data` replays
  deterministically from committed `data/raw/`); add to the real row list in `format_tables.py:115`.
- **`run_baseline.py` lane**: optionally print BAPG in the paper-style single-cell path too;
  if so, fix its seed gap first (add `seed` to `build_dataset_config` + a `--seed` flag).
- **Larger dense run**: a fresh all-methods seeded run with more dense pairs if n=10 CIs are
  too wide to say anything.

---

## Execution log (2026-07-23)

Decisions taken by the author: D1 in-repo; D2 core dep (`pot==0.9.7` locked); **D3 `bapg_proj`
only** (no FAQ-refined row); D4 post-hoc merge; D5 paper defaults; D6 synthetic only; D7 keep
dense n=10. Additionally: **no solver re-runs for validation** — the D_cx cross-check gate was
dropped (author instruction) after it spuriously aborted on macOS-vs-CLEPS BLAS drift (43%
exact per-sample match but |Δmean acc| = 0.0014); the noise-0 **edge-count fingerprint**
(solver-free) is the only gate, and it verified replay identity exactly: 30/30 sparse, 10/10
dense committed `faq_dcx` nce values equal the replayed pairs' edge counts. On regular the
fingerprint is uninformative (D_cx collapse), replay identity rests on the verbatim seeding
call order.

Pilot findings (Phase C):
- POT's `tol` is an **absolute** plan-change norm checked every 10 iterations
  (`ot/gromov/_bregman.py`), not the paper's relative rule — harmless at ε=0.1.
- Paper defaults reproduce: sparse@0 acc 0.988, dense@0–0.15 acc 1.000 on pilot pairs, and a
  Gaussian-random-partition positive control (the paper's own graph type, n=500) at 1.000.
- ε sensitivity on sparse mid-noise is real but non-directional (12-pair check of 0.1 vs 1e-3:
  neither dominates; per-pair variance is huge either way). ε=0.1 kept per D5.
- Regular family flat-degenerate at every ε tried (acc ≈ 0.006 pilot): uniform degrees give
  the multiplicative update no first-order signal from the flat start `G0 = pqᵀ`.

Final grid (21/21 cells, merged into `repro/results/repro_seed0.jsonl`, key `bapg_proj`):
- **sparse**: 0.98/994 → 0.89/893 (0.05) → 0.71/772 (0.1) → 0.35/624 (0.15) → 0.009/520
  (0.35). Degrades earlier than every FAQ-decoded method — below FAQ(D_cx) from 0.05 on.
- **dense**: acc 1.00 through p=0.2 (nce = max), transition at 0.25 (0.32/8832) — nearly
  identical to FAQ(D_cx) (0.32/8854); 0.13@0.3 vs chain 0.22, FGNN-FAQ 0.81.
- **regular**: 0.002/≈50 flat — cell-for-cell equal to Proj(D_cx)'s collapsed row.
- Failure overlap (threshold acc>0.5): ChFGNN-FAQ solves 192 pairs BAPG-GW misses; BAPG-GW
  solves exactly **1** pair the chain misses (dense@0.3) — near-strict dominance.
- Runtime: full grid ≈ 50 min single-threaded CPU (sparse ≈ 10 s/pair at 2000 iters, dense
  and regular converge to fixed points in well under 1 s/pair).

Timing convention (author request, 2026-07-23): every per-sample evaluation stores a
`time` array (wall seconds per pair, 2 dp) beside `acc`/`nce` in the method's JSONL
entry, and progress lines print mean s/pair. `add_bapg.py` and `add_fgwalign.py` both
record it; the committed `bapg_proj` entries gained `time` via a `--force` rerun on
2026-07-24 that reproduced every committed acc/nce byte-identically (end-to-end
determinism check of the replay machinery).

## FGWAlign extension (2026-07-24)

FGWAlign (Tang et al., PVLDB 18(11), 2025 — same group as BAPG) added as a second
external baseline, key `fgwalign`, via `repro/add_fgwalign.py`. Upstream repo has **no
license** → not vendored; the driver imports from a user-provided clone (core needs only
torch/pot/numpy). Protocol: authors' defaults (patience=15, topk=5, full solver) with
`sparse=True` — the dense path overflows float32 at n=500 (exp(-cost/0.01) on the
complement-graph term → NaN → segfault inside POT's C EMD; sparse mode drops that term,
same optimum over permutations). Light variant disqualified (acc 0.03 vs 0.45 at
sparse@0.1). Stochastic solver → `seed_everything(seed)` before every pair.

Results: **statistically the same solver as BAPG-GW on these regimes** — identical
solved/failed status on 469/470 pairs; both rescue the same dense@0.3 pair the chain
misses; chain solves 191 pairs FGWAlign cannot. Sparse/dense rows match BAPG's within
noise. Only separation: regular-family nce ≈ 820/2500 (vs ≈ 50 for BAPG/Proj(D_cx)) —
the GED objective salvages common edges at chance node accuracy. Runtime (default
protocol, 1 CPU core): ~70 s/pair sparse, ~150–210 s/pair dense, ~85–95 s/pair regular
(≈ 4–4.6 h per family, run in parallel) vs BAPG's ~10 / <1 / <1 s/pair.

## Risks / gotchas (carry into implementation)

1. **T orientation**: LAP on `-T.T`, not `-T` — the isomorphism test locks it; write it first.
2. **EdgeSwap replay**: any extra global-RNG consumption before `get_data` in the merge script
   silently changes the regular-family pairs — the Phase D validation gate is the defense.
3. **`main()` truncates `--out`**: never target the committed jsonl directly.
4. **All-or-nothing**: `bapg_*` keys must cover every ER-Reg cell or `make tables`/`make
   overlap` crash.
5. **ε failure mode**: too-small ε → NaN plan; the retry-then-identity-fallback policy keeps
   grid runs alive. Don't interpret `gw_dist`/negative losses — use `T` only.
6. **Expected regular degeneracy**: near-random BAPG on d-regular graphs is a plausible,
   reportable outcome (same mechanism as the documented D_cx collapse) — pre-decide the README
   wording, don't debug it as a failure.
7. **Original-paper protocol difference**: the ICLR paper uses row-argmax extraction; our table
   rows use LAP/FAQ (stronger). State this explicitly in the README to keep the comparison fair.
