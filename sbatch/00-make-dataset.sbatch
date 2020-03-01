#!/bin/bash

#BATCH --job-name=00-dataset
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=24GB

#SBATCH --output=/home/hh2263/wave2shape/slurm/slurm_dataset.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29


python /home/hh2263/wave2shape/src/make_dataset.py
