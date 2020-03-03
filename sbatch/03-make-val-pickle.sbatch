#!/bin/bash

#BATCH --job-name=make-val-pickle
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=62GB

#SBATCH --output=/home/hh2263/wave2shape/slurm/slurm_make_pickle_812.out   


module purge

module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29


python /home/hh2263/wave2shape/src/make_pickle.py 8 1 2
