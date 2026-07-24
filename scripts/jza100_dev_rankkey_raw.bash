#!/bin/bash

# Degree-normalized ranking ablation — CONTROL arm (raw ranking), qos_dev.
# Same reduced budget as the degnorm arm so the two are directly comparable
# (the released chain used a larger budget; comparisons are ablation-internal).
# Checkpoints save per link, so the 2 h dev kill loses only the in-flight link.

#SBATCH --job-name=rankkey_raw_s022
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
srun python commander.py dataset=sparse dataset.noise=0.22 \
    pipeline.rank_key=raw pipeline.L=10 \
    pipeline.path_models=${SLURM_JOB_NAME} \
    model.in_features=256 training.batch_size=6 training.epochs=100 \
    hydra/run=cluster root_dir=$SCRATCH
