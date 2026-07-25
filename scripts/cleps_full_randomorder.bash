#!/bin/bash

# Random-order ablation — FULL-BUDGET sparse training on CLEPS (rtx8000).
# rtx8000 (46 GiB, non-preemptible for us) mirrors the released recipe (L=15,
# epochs=400 cap, in_features=256, batch 6, noise 0.22) with random inter-link
# order. Uses the venv python directly (NOT `uv run`) so the working
# torch 2.8+cu128 stack is never re-synced/clobbered; source edits are live via
# cwd. Outputs to /scratch (HOME quota is small).

#SBATCH --job-name=randomorder_full_cleps
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/lelarge/GitHub/chaining-gnn-graph-alignment
PY=$REPO/.venv/bin/python
cd $REPO
mkdir -p /scratch/lelarge/experiments-gnn-gap/data

set -x
srun $PY commander.py dataset=sparse dataset.noise=0.22 \
    pipeline.random_order=true pipeline.L=15 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=6 training.wandb=false \
    hydra/run=cluster root_dir=/scratch/lelarge
RC=$?
[ "$RC" -eq 0 ] && echo "DONE_OK" || echo "FAIL_RC=$RC"
exit $RC
