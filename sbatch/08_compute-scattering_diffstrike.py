import os
import sys

sys.path.append("../src")


# Define constants.
folds = ["test", "train", "val"]
Js = [6, 8, 10, 12, 14]
orders = [1, 2]
script_name = os.path.basename(__file__)
script_path = os.path.join("..", "..", "..", "src", script_name)


# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)


# Loop over folds: training, test, and validation.
for fold in folds:

    # Loop over scales.
    for J in Js:

        # Loop over scattering orders.
        for order in orders:

            script_path_with_args = " ".join(
                [script_path, str(fold), str(J), str(order)])
            job_name = "_".join([
                script_name[:2],
                "fold-" + str(fold), "J-" +  str(J).zfill(2), "order-" + str(order)
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
                f.write("#SBATCH --cpus-per-task=10\n")
                f.write("#SBATCH --time=15:00:00\n")
                f.write("#SBATCH --mem=32GB\n")
                f.write("#SBATCH --output=" +\
                    "../slurm/" + job_name + "_%j.out\n")
                f.write("\n")
                f.write("module purge\n")
                f.write("source activate w2s\n")
                f.write("\n")
                f.write("# The first argument is the fold: training or validation.\n")
                f.write("# The second argument is the quality factor.\n")
                f.write("# The third argument is the scattering order.\n")
                f.write("python " + script_path_with_args)


# Open shell file.
file_path = os.path.join(sbatch_dir, script_name[:2] + ".sh")

with open(file_path, "w") as f:
    # Print header.
    f.write("# This shell script computes scattering features for various hyperparmeter settings.\n")
    f.write("\n")

    # Loop over folds: training and validation.
    for fold in folds:

        # Loop over scales.
        for J in Js:

            # Loop over scattering orders.
            for order in orders:

                # Define job name.
                job_name = "_".join([
                    script_name[:2],
                    "fold-" + str(fold),
                    "J-" + str(J).zfill(2),
                    "order-" + str(order)])
                sbatch_str = "sbatch " + job_name + ".sbatch"

                # Write SBATCH command to shell file.
                f.write(sbatch_str + "\n")
        f.write("\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)
