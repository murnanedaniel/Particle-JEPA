#!/bin/bash

#SBATCH -A m3443 -q regular
#SBATCH -C gpu
#SBATCH -t 6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -o logs/%x-%j.out
#SBATCH -J GTrack-train
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH --module=gpu,nccl-2.18
#SBATCH --signal=SIGUSR1@90

# Setup
mkdir -p logs

eval "$(conda shell.bash hook)"

source activate gtrack

export SLURM_CPU_BIND="cores"
echo -e "\nStarting training\n"

# Multiple GPU training
srun -u python run.py $@