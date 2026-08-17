#!/bin/bash

# Random-order ablation — h100 FAST-TRACK (batch 6 directly; ~2x faster than the
# batch-3 rtx8000 path). Duplicates the running rtx8000 random-order job to get
# a result sooner; h100 is PREEMPTIBLE (--requeue), the rtx8000 run (alloc
# 5140330) is the non-preemptible backstop — whichever finishes first wins, the
# other is cancelled. Distinct path_models so dirs / wandb never collide.

#SBATCH --job-name=randomorder_full_h100
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
    pipeline.random_order=true pipeline.L=15 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=6 training.epochs=100 \
    training.wandb=true \
    hydra/run=default root_dir=/scratch/lelarge
RC=$?
[ "$RC" -eq 0 ] && echo "DONE_OK" || echo "FAIL_RC=$RC"
exit $RC
