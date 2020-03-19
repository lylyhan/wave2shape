import os
import sys

sys.path.append("../src")


# Define constants.
trials = [1,2,3,4,5,6,7,8,9,10]
script_name = os.path.basename(__file__)
script_path = os.path.join("..", "..", "..", "src", script_name)


# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)

for trial in trials:

    job_name = "_".join([
        script_name[:2],
        "trial-" + str(trial)
    ])
    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)

    # Generate file.
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + script_name[:2] + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=4\n")
        f.write("#SBATCH --time=10:00:00\n")
        f.write("#SBATCH --mem=62GB\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --output=" +\
            "../slurm/" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("module load cuda/9.0.176\n")
        f.write("module load cudnn/9.0v7.3.0.29\n")
        f.write("module load ffmpeg/intel/3.2.2\n")
        f.write("\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 6 1 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 6 2 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 8 1 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 8 2 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 10 1 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 10 2 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 12 1 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 12 2 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 14 1 "+str(trial)+"\n")
        f.write("python /home/hh2263/wave2shape/src/06_train.py 14 2 "+str(trial)+"\n")
       



# Open shell file.
file_path = os.path.join(sbatch_dir, script_name[:2] + ".sh")

with open(file_path, "w") as f:
    # Print header.
    f.write("# This shell script computes scattering features for various hyperparmeter settings.\n")
    f.write("\n")

    # Loop over folds: training and validation.
    for trial in trials:

    # Define job name.
        job_name = "_".join([
            script_name[:2],
            "trial-" + str(trial)])
        sbatch_str = "sbatch " + job_name + ".sbatch"

        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")
        f.write("\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)
