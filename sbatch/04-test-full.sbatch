#!/bin/bash

#BATCH --job-name=test_all
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/hh2263/wave2shape/slurm/slurm_test_full_w_loop_.out   


module purge

module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

python /home/hh2263/wave2shape/src/preliminary_test_alt.py 12 128 512 8 1 2 10
