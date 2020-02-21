#!/bin/bash

#BATCH --job-name=00-dataset
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=24GB
#SBATCH --output=../slurm/slurm_dataset.out

module purge

# The first argument is the name of the recording unit.
# The second argument is the name of the trial.
python ../src/make_dataset.py
