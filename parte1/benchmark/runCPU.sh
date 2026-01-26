#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=%x_%j.out
#SBATCH -t 00:05:00
#SBATCH -N 1	#Number of nodes
#SBATCH --exclusive
#SBATCH --gres=gpu:T4:1

module load cuda/11.8.0 gnu12/12.2.1 python/3.12.9 openssl/1.1.1w
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_NTASKS=$SLURM_NTASKS
id && hostname
date

echo "============================= CPU =================================="
CPU_THREADS=$SLURM_NTASKS python3 benchmark_linear_algebra.py --device cpu
echo "===================================================================="

date
