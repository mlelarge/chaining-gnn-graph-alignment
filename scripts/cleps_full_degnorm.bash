#!/bin/bash

# Degree-normalized ablation — FULL-BUDGET sparse training on CLEPS (rtx8000).
# Runs in PARALLEL with the random-order chain (a 16h batch-3 chain does not
# fit twice in one 20h allocation, and the allocation TimeLimit cannot be
# extended). Non-preemptible rtx8000, gpu009 excluded. Effective batch 6 via
# micro-batch 3 x accumulate 2. wandb online (project = job name).

#SBATCH --job-name=degnorm_full_cleps
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --exclude=gpu009
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
    pipeline.rank_key=degree_normalized pipeline.L=15 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=3 training.accumulate_grad_batches=2 \
    training.wandb=true \
    hydra/run=default root_dir=/scratch/lelarge
RC=$?
[ "$RC" -eq 0 ] && echo "DONE_OK" || echo "FAIL_RC=$RC"
exit $RC
