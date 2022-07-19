"""
run one trial experiment, produce prediction on test set, save as npy file
"""
import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append("../src")
sys.path.append("/home/han/kymatio-jtfs/")
sys.path.append("../scripts")

import train
import features
import data_generator
import cnn

import pandas as pd
import librosa
import pescador
from tqdm import tqdm

ftype = "cqt"
J = 8
Q = 16
logscale = 1e-3
param = "alpha"
activation = "linear_prevsummse"
loss = "ploss"
batchsize = 64
n_epoch = 30
steps_per_epoch = 50
is_normalize = True
is_multitask = True
lr = 0.01

if __name__ == "__main__":
    
    val_loss,train_loss = train.run_train({"type":ftype,"J":J,"Q":Q,"param":param},
                                                        trial=0,
                                                        logscale=logscale,
                                                        is_normalize=is_normalize,
                                                        is_multitask=is_multitask,
                                                        activation=activation,
                                                        loss=loss,
                                                        batch_size=batchsize,
                                                        n_epoch=n_epoch,
                                                        lr=lr,
                                                        steps_per_epoch=steps_per_epoch,
                                                        predict_mode=True)
    
    print("finished training and predicting, ",val_loss," ",train_loss)

    
    
    
    
    