#!/bin/bash

#BATCH --job-name=00-dataset
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=24GB
#SBATCH --output=../slurm/slurm_dataset.out

module purge
module load cuda/9.0.176


cd /home/hh2263/wave2shape/src/
python make_dataset.py
