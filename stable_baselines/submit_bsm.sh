#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40

source /home/mjad1g20/.bashrc
source activate phenogame

python bsm_train.py
