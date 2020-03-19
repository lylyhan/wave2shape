#!/bin/bash

#BATCH --job-name=05
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/05_fold-train_J-06_order-2_%j.out

module purge

# The first argument is the fold: training or validation.
# The second argument is the quality factor.
# The third argument is the scattering order.
python ../../../src/05_compute-scattering.py train 6 2
