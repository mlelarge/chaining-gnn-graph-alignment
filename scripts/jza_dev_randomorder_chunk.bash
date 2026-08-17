#!/bin/bash

# Random-order full chain on JZ via qos_dev, SPLIT into 2-link chunks.
# Each 2h dev job trains up to chunk_size new links (commander_chunk resumes
# from the checkpoints already on disk). A controller resubmits this until all
# 15 links exist. Fast-starting dev queue instead of waiting for a long t3 slot.

#SBATCH --job-name=ro_chunk_s022
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

set -x
ROOTDIR=${SCRATCH:-/lustre/fsn1/projects/rech/tdm/uuz44ie}
srun python commander_chunk.py dataset=sparse dataset.noise=0.22 \
    pipeline.random_order=true pipeline.L=15 +pipeline.chunk_size=2 \
    pipeline.path_models=ro_chunk_s022 \
    model.in_features=256 training.batch_size=6 training.epochs=100 training.wandb=false \
    hydra/run=cluster root_dir=$ROOTDIR
