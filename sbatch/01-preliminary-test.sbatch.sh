#!/bin/bash

#BATCH --job-name=preliminary-test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/hh2263/wave2shape/slurm/slurm_test.out   


module purge
source activate birdvox
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

python /home/hh2263/wave2shape/src/preliminary_test.py