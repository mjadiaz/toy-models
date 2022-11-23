#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10

source /home/mjad1g20/.bashrc
source activate phenogame

python ddpg_sb3.py
