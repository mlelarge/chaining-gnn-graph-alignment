#!/bin/bash

# Random-order ablation — DEV-budget sparse training (qos_dev, quick trajectory).
# Lower L=5, epochs=100 per link (same per-link budget as the rankkey dev arms,
# so directly comparable to them). Tests whether a chain trained with a RANDOM
# inter-link order still improves link-over-link, or stays at single-FGNN level.

#SBATCH --job-name=randomorder_dev_s022
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
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
    pipeline.random_order=true pipeline.L=5 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=6 training.epochs=100 \
    hydra/run=cluster root_dir=$ROOTDIR
