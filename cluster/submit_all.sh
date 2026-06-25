#!/bin/bash
# Launch the full seeded reproduction on CLEPS — one SLURM job per experiment,
# each on its own node. RUN THIS ON THE CLEPS LOGIN NODE (it has internet; the
# compute nodes may not, so we build the env + fetch checkpoints here first).
#
#   cd ~/GitHub/chaining-gnn-graph-alignment
#   bash cluster/submit_all.sh
#
# Overridable via env: REPO, CKPT, SEED, NUM, PARTITION.
set -euo pipefail

REPO="${REPO:-$HOME/GitHub/chaining-gnn-graph-alignment}"
CKPT="${CKPT:-$REPO/checkpoints}"
SEED="${SEED:-0}"
NUM="${NUM:-100}"
PARTITION="${PARTITION:-cpu_homogen}"   # CLEPS CPU partition (override if needed)

cd "$REPO"

# 1. Environment — built once on the login node; .venv lives on shared /home so
#    every compute node sees it (the jobs run `.venv/bin/python`, no network).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "Syncing environment (uv sync) ..."
uv sync

# 2. Pretrained checkpoints — fetched here (internet) into the shared cache.
echo "Pre-fetching pretrained releases into $CKPT ..."
uv run python cluster/prefetch_releases.py "$CKPT"

# 3. Submit one job per experiment. Each gets its own node; dense gets the most
#    wall-time. SLURM distributes them across free nodes of $PARTITION.
declare -A WALLTIME=(
    [regular]=12:00:00
    [real]=1-00:00:00
    [sparse]=1-12:00:00
    [dense]=4-00:00:00
)

echo "Submitting jobs to partition '$PARTITION' (seed=$SEED, num-examples=$NUM):"
for EXP in regular real sparse dense; do
    jid=$(sbatch --parsable \
        --job-name="chgnn-$EXP" \
        --partition="$PARTITION" \
        --time="${WALLTIME[$EXP]}" \
        --export=ALL,EXP="$EXP",REPO="$REPO",CKPT="$CKPT",SEED="$SEED",NUM="$NUM" \
        cluster/cleps_repro.sbatch)
    echo "  $EXP -> job $jid (log: repro_chgnn-${EXP}_${jid}.out)"
done

echo ""
echo "Watch:    squeue -u $USER"
echo "Results:  repro_{regular,real,sparse,dense}.jsonl  (one JSON record per table cell)"
echo "When all four are done, collect the four .jsonl files."
