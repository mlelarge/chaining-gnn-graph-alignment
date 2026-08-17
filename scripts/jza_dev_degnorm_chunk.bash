#!/bin/bash

# Degree-normalized full chain on JZ via qos_dev, SPLIT into 2-link chunks
# (commander_chunk resumes from on-disk checkpoints). Sibling of
# jza_dev_randomorder_chunk.bash; a controller resubmits until 15 links exist.

#SBATCH --job-name=dn_chunk_s022
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
    pipeline.rank_key=degree_normalized pipeline.L=15 +pipeline.chunk_size=2 \
    pipeline.path_models=dn_chunk_s022 \
    model.in_features=256 training.batch_size=6 training.epochs=100 training.wandb=false \
    hydra/run=cluster root_dir=$ROOTDIR
