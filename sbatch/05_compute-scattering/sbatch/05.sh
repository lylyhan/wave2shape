# This shell script computes scattering features for various hyperparmeter settings.

sbatch 05_val_1_1.sbatch
sbatch 05_val_1_2.sbatch
sbatch 05_val_2_1.sbatch
sbatch 05_val_2_2.sbatch

sbatch 05_train_1_1.sbatch
sbatch 05_train_1_2.sbatch
sbatch 05_train_2_1.sbatch
sbatch 05_train_2_2.sbatch

