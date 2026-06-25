# Running the seeded reproduction on CLEPS

One SLURM job per experiment (`sparse`, `dense`, `regular`, `real`) — each on its
own node. The login node (internet) builds the env + fetches the pretrained
checkpoints once; the compute nodes run **offline** from the shared `.venv` and
`./checkpoints` (`/home` is shared across CLEPS nodes).

CLEPS CPU partitions (both allow up to 1 week): `cpu_homogen` (node001-020, 32
cores / 192 GB) — the default here — and `cpu_devel` (node021-056, the cluster
default). Override with `PARTITION=...`.

## 1. Get the code onto CLEPS

`release/repro` is not on GitHub yet, so copy the working tree up (matches your
`synch_cleps.sh`). **From your laptop:**

```bash
rsync -azh \
  --exclude-from=/Users/lelarge/Recherche/match-gnn/chaining-gnn-graph-alignment/.gitignore \
  --exclude='.git/' --exclude='tests/fixtures/' \
  /Users/lelarge/Recherche/match-gnn/chaining-gnn-graph-alignment \
  lelarge@cleps.inria.fr:/home/lelarge/GitHub/
```

This includes `data/raw/` (committed) and `cluster/`, and excludes `.venv`,
`data/prepared/`, `checkpoints/`, fixtures. (Alternative: `git push -u origin
release/repro` from the laptop, then `git clone -b release/repro
git@github.com:mlelarge/chaining-gnn-graph-alignment.git
~/GitHub/chaining-gnn-graph-alignment` on CLEPS.)

## 2. Launch (on the CLEPS login node)

```bash
ssh lelarge@cleps.inria.fr
cd ~/GitHub/chaining-gnn-graph-alignment
bash cluster/submit_all.sh
```

This installs `uv` if needed, runs `uv sync`, pre-fetches all releases into
`./checkpoints`, then submits the four jobs. Useful overrides:

```bash
SEED=0 NUM=100 PARTITION=cpu_homogen bash cluster/submit_all.sh   # defaults
NUM=2 bash cluster/submit_all.sh                                  # quick env sanity run
```

Sanity-check the env on the login node first (optional, ~minutes):

```bash
uv run python -m repro.reproduce_results --family sparse --noises 0.1 \
  --seed 0 --num-examples 2 --N-max 5 --out /tmp/smoke.jsonl && cat /tmp/smoke.jsonl
```

## 3. Collect

```bash
squeue -u $USER                       # watch
cat repro_*.jsonl                     # results (one JSON record per table cell)
```

Each job writes `repro_<exp>.jsonl`. When all four finish, send the four files
back — they get turned into the README results tables (replacing the
paper-transcribed numbers, with a "reproduced, seed=0, N=100" note).

**Notes.** `dense` is the long pole (multi-hour). Jobs are seeded end-to-end, so
the numbers are reproducible (close to, not identical to, the unseeded paper).
BLAS uses the allocated cores; for strict bit-determinism across machines set
`OMP_NUM_THREADS=1` (slower).
