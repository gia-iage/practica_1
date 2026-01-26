#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=%x_%j.out
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:T4:1

module load cuda/11.8.0 gnu12/12.2.1 python/3.12.9 openssl/1.1.1w
echo SLURM_JOBID=$SLURM_JOBID
id && hostname
date

echo "===================================================================="
python3 --version
nvidia-smi
/share/apps/utils/deviceQuery
sleep 100
echo "===================================================================="

date
