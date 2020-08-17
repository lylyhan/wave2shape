#!/bin/bash

#BATCH --job-name=07
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/07_trial-2_%j.out

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

python /home/hh2263/wave2shape/src/06_train.py 6 1 3 2
python /home/hh2263/wave2shape/src/06_train.py 6 2 3 2
python /home/hh2263/wave2shape/src/06_train.py 8 1 3 2
python /home/hh2263/wave2shape/src/06_train.py 8 2 3 2
python /home/hh2263/wave2shape/src/06_train.py 10 1 3 2
python /home/hh2263/wave2shape/src/06_train.py 10 2 3 2
python /home/hh2263/wave2shape/src/06_train.py 12 1 3 2
python /home/hh2263/wave2shape/src/06_train.py 12 2 3 2
python /home/hh2263/wave2shape/src/06_train.py 14 1 3 2
python /home/hh2263/wave2shape/src/06_train.py 14 2 3 2
