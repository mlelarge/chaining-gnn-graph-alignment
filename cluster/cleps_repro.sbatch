#!/bin/bash
# CLEPS SLURM job: reproduce one experiment block of the paper tables (seeded).
# One job == one node. Submitted by cluster/submit_all.sh (which sets EXP etc.
# and may override --partition/--time on the sbatch command line).
#
# Runs OFFLINE: uses the repo's .venv and the pre-fetched ./checkpoints, both of
# which submit_all.sh prepares on the login node (compute nodes may lack internet).
#SBATCH --job-name=chgnn-repro
#SBATCH --partition=cpu_homogen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=repro_%x_%j.out

set -euo pipefail

EXP="${EXP:?set EXP=sparse|dense|regular|real}"
REPO="${REPO:-$HOME/GitHub/chaining-gnn-graph-alignment}"
CKPT="${CKPT:-$REPO/checkpoints}"
SEED="${SEED:-0}"
NUM="${NUM:-30}"
NMAX="${NMAX:-15}"

cd "$REPO"

# Pin BLAS / torch threads to the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

PY=.venv/bin/python
echo "[$(date)] node=$(hostname) EXP=$EXP seed=$SEED num=$NUM cpus=$OMP_NUM_THREADS"

SEL=(--family "$EXP")
[ "$EXP" = "real" ] && SEL=(--real)

"$PY" -m repro.reproduce_results "${SEL[@]}" \
    --seed "$SEED" --num-examples "$NUM" --N-max "$NMAX" \
    --checkpoint-dir "$CKPT" \
    --data-dir "./data/prepared_${EXP}" \
    --out "repro_${EXP}.jsonl"

echo "[$(date)] DONE $EXP -> repro_${EXP}.jsonl"
