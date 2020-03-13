#!/bin/bash

#BATCH --job-name=05
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --output=../slurm/05_fold-test_J-08_order-2_%j.out

module purge
source activate w2s

# The first argument is the fold: training or validation.
# The second argument is the quality factor.
# The third argument is the scattering order.
python ../../../src/05_compute-scattering.py test 8 2