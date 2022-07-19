import numpy as np
import pandas
import random
import ftm
import soundfile as sf
import os
#from kymatio import Scattering1D

df_train = pandas.read_csv("../notebooks/train_param.csv") 
df_val = pandas.read_csv("../notebooks/val_param.csv") 
df_test = pandas.read_csv("../notebooks/test_param.csv") 

val_params = df_val.values
train_params = df_train.values
test_params = df_test.values

n_samp_val,n_param_val = val_params.shape
n_samp_test,n_param_val = test_params.shape
n_samp_train,n_param_val = train_params.shape


mode = 10
sr = 22050

for i in range(n_samp_val):
    omega,tau,p,D,alpha = val_params[i,1:-1]
    y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
    y = y/ max(y)
    path_out = "/scratch/hh2263/drum_data/val/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    filename = os.path.join(path_out,str(val_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)


print("finished validation set!!")

for i in range(n_samp_test):
    omega,tau,p,D,alpha = test_params[i,1:-1]
    y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
    y = y/ max(y)
    path_out = "/scratch/hh2263/drum_data/test/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    filename = os.path.join(path_out,str(test_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)
    
    
print("finished test set!!")

for i in range(n_samp_train):
    omega,tau,p,D,alpha = train_params[i,1:-1]
    if not os.path.exists("/scratch/hh2263/drum_data/train/"+str(train_params[i,0])+"_sound.wav"):
        y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
        y = y/ max(y)
        path_out = "/scratch/hh2263/drum_data/train/"
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        filename = os.path.join(path_out,str(train_params[i,0])+"_sound.wav")
        sf.write(filename, y, sr)
    

print("finished train set!!")






