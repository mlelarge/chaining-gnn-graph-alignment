#!/bin/bash

# Degree-normalized ablation — h100 FAST-TRACK (batch 6 directly; 94 GiB fits
# it, no accumulation needed, ~2x faster than the batch-3 rtx8000 path). h100
# is PREEMPTIBLE, so --requeue; the rtx8000 degnorm job is the non-preemptible
# backstop (dedupe: whichever starts first, cancel the other). Distinct
# path_models so wandb / result dirs never collide with the rtx8000 run.

#SBATCH --job-name=degnorm_full_h100
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
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
    model.in_features=256 training.batch_size=6 \
    training.wandb=true \
    hydra/run=default root_dir=/scratch/lelarge
RC=$?
[ "$RC" -eq 0 ] && echo "DONE_OK" || echo "FAIL_RC=$RC"
exit $RC
