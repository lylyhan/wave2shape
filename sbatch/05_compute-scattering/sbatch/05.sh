# This shell script computes scattering features for various hyperparmeter settings.

sbatch 05_fold-test_J-06_order-1.sbatch
sbatch 05_fold-test_J-06_order-2.sbatch
sbatch 05_fold-test_J-08_order-1.sbatch
sbatch 05_fold-test_J-08_order-2.sbatch
sbatch 05_fold-test_J-10_order-1.sbatch
sbatch 05_fold-test_J-10_order-2.sbatch
sbatch 05_fold-test_J-12_order-1.sbatch
sbatch 05_fold-test_J-12_order-2.sbatch
sbatch 05_fold-test_J-14_order-1.sbatch
sbatch 05_fold-test_J-14_order-2.sbatch

sbatch 05_fold-train_J-06_order-1.sbatch
sbatch 05_fold-train_J-06_order-2.sbatch
sbatch 05_fold-train_J-08_order-1.sbatch
sbatch 05_fold-train_J-08_order-2.sbatch
sbatch 05_fold-train_J-10_order-1.sbatch
sbatch 05_fold-train_J-10_order-2.sbatch
sbatch 05_fold-train_J-12_order-1.sbatch
sbatch 05_fold-train_J-12_order-2.sbatch
sbatch 05_fold-train_J-14_order-1.sbatch
sbatch 05_fold-train_J-14_order-2.sbatch

sbatch 05_fold-val_J-06_order-1.sbatch
sbatch 05_fold-val_J-06_order-2.sbatch
sbatch 05_fold-val_J-08_order-1.sbatch
sbatch 05_fold-val_J-08_order-2.sbatch
sbatch 05_fold-val_J-10_order-1.sbatch
sbatch 05_fold-val_J-10_order-2.sbatch
sbatch 05_fold-val_J-12_order-1.sbatch
sbatch 05_fold-val_J-12_order-2.sbatch
sbatch 05_fold-val_J-14_order-1.sbatch
sbatch 05_fold-val_J-14_order-2.sbatch

