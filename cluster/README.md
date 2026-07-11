# Running the seeded reproduction on CLEPS

One SLURM job per experiment (`sparse`, `dense`, `regular`, `real`) — each on its
own node. The login node (internet) builds the env + fetches the pretrained
checkpoints once; the compute nodes run **offline** from the shared `.venv` and
`./checkpoints` (`/home` is shared across CLEPS nodes).

CLEPS CPU partitions (both allow up to 1 week): `cpu_homogen` (node001-020, 32
cores / 192 GB) — the default here — and `cpu_devel` (node021-056, the cluster
default). Override with `PARTITION=...`.

## 1. Get the code onto CLEPS

`release/repro` is not on GitHub, so copy the working tree up (like
`synch_cleps.sh`). Syncing into a **fresh directory** leaves any existing clone
untouched. **From your laptop** (trailing slash → contents go *into* the dir; no
`--delete`, so the cluster's `.venv`/`checkpoints` survive a re-sync):

```bash
rsync -azh --exclude-from=.gitignore --exclude='.git/' --exclude='tests/fixtures/' \
  /Users/lelarge/Recherche/match-gnn/chaining-gnn-graph-alignment/ \
  lelarge@cleps.inria.fr:/home/lelarge/GitHub/chaining-gnn-repro/
```

This includes `data/raw/` and `cluster/`, and excludes `.venv`, `data/prepared/`,
`checkpoints/`, fixtures.

## 2. Launch (on the CLEPS login node)

```bash
ssh lelarge@cleps.inria.fr
cd ~/GitHub/chaining-gnn-repro
REPO=$PWD bash cluster/submit_all.sh
```

Installs `uv` if needed, runs `uv sync`, pre-fetches the releases into
`./checkpoints`, then submits four jobs (one per node). Defaults: `SEED=0`,
`NUM=30` synthetic pairs/cell (`dense` uses 10), `NMAX=15` refinement cap.
Overrides:

```bash
NUM=10 REPO=$PWD bash cluster/submit_all.sh          # ~3x faster, slightly noisier averages
SEED=1 NMAX=20 REPO=$PWD bash cluster/submit_all.sh
PARTITION=cpu_devel REPO=$PWD bash cluster/submit_all.sh
```

Optional env smoke-test on the login node first (~1 min):

```bash
uv run python -m repro.reproduce_results --family sparse --noises 0.1 \
  --seed 0 --num-examples 2 --N-max 5 --out /tmp/smoke.jsonl && cat /tmp/smoke.jsonl
```

## 3. Monitor & collect

```bash
squeue --me
wc -l repro_*.jsonl 2>/dev/null                 # rows -> regular 5 / real 6 / sparse 8 / dense 8
tail -3 repro_chgnn-regular_<jobid>.out         # live progress (ls repro_*.out for jobids)
grep -iE 'error|oom|killed|traceback' repro_chgnn-*_<jobids>.out   # should stay empty
```

Each job writes `repro_<exp>.jsonl` (the driver truncates it at start, so re-runs
are clean). When the four finish, send them back — they become the README tables
(replacing the paper-transcribed numbers, with a "reproduced, seed=0, num=30/10"
note).

## Notes

- **Pace.** The cost per cell is the 11-model chain eval + FAQ on n=500 — each is
  a 2-FWL FGNN forward (built for GPU), so ~tens of minutes/cell on CPU.
  `regular`/`real` ~hours, `sparse` longer, `dense` the long pole (day-ish at
  `num=10`). All within the wall-times.
- **Reproducible, not paper-identical.** Seeded end-to-end (close to, not equal
  to, the unseeded paper). On *regular* graphs the convex baselines
  (`Proj/FAQ D_cx`) are degenerate by design, so their `nce` is high-variance and
  only approximate — the ChFGNN rows reproduce cleanly. Sparse/dense `D_cx` are
  non-degenerate and track the paper closely.
- BLAS uses the allocated cores; for strict bit-determinism across machines set
  `OMP_NUM_THREADS=1` (slower).
