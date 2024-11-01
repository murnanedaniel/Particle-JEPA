#!/bin/bash

#SBATCH -A m3443 -q regular
#SBATCH -C gpu
#SBATCH -t 60:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J JEA-train
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --signal=SIGUSR1@90

# This is a generic script for submitting training jobs to Perlmutter GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
eval "$(conda shell.bash hook)"

conda activate jepa

export SLURM_CPU_BIND="cores"
export WANDB__SERVICE_WAIT=300
echo -e "\nStarting sweeps\n"

# Single GPU training
srun -u python run.py $@
