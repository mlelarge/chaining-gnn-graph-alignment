#!/bin/bash

# Random-order ablation — FULL-BUDGET sparse training (t3). Identical to the
# released recipe (L=15, epochs=400 cap, in_features=256, batch 6, noise 0.22)
# except the inter-link feedback is ordered RANDOMLY instead of by score. The
# released score-order chain is the matched-budget control; link 0 (a single
# FGNN, no feedback) is the lower baseline.

#SBATCH --job-name=randomorder_full_s022
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=tdm@a100

module purge
conda deactivate

module load arch/a100
module load pytorch-gpu/py3/2.3.0
wandb offline

set -x
ROOTDIR=${SCRATCH:-/lustre/fsn1/projects/rech/tdm/uuz44ie}
srun python commander.py dataset=sparse dataset.noise=0.22 \
    pipeline.random_order=true pipeline.L=15 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=6 \
    hydra/run=cluster root_dir=$ROOTDIR
