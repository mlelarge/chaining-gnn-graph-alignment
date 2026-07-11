#!/bin/bash

#SBATCH --job-name=jza100_ER_500_4_24_r256 # name of job
#SBATCH --output=%x_%j.out  # output file (%j = job ID)
#SBATCH --error=%x_%j.err # error file (%j = job ID)
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 1 nodez
#SBATCH --ntasks=1 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:1 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=12:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=tdm@a100 # A100 accounting

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load arch/a100
module load pytorch-gpu/py3/2.3.0
wandb offline

set -x # activate echo of launched commands
srun python commander.py dataset=sparse pipeline.path_models=${SLURM_JOB_NAME} hydra/run=cluster training.wandb=Yes model.in_features=256 training.batch_size=6 dataset.noise=0.24