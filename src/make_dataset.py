import numpy as np
import pandas as pd
import random
import ftm
import soundfile as sf
import os
#from kymatio import Scattering1D

df_train = pandas.read_csv("train_param.csv") 

mode = 10
sr = 44100
train_params = df_train.values
n_samp,n_param = params.shape
for i in range(n_samp):
    omega,tau,p,D,alpha = train_params[i,1:-1]
    y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
    y = y/ max(y)
    path_out = "/scratch/hh2263/drum_data/train/"
   	if not os.path.exists(path_out):
   		os.mkdir(path_out)
    filename = os.path.join(path_out,str(train_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)
    



df_test = pandas.read_csv("test_param.csv") 
test_params = df_test.values

for i in range(n_samp):
    omega,tau,p,D,alpha = test_params[i,1:-1]
    y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
    y = y/ max(y)
    path_out = "/scratch/hh2263/drum_data/test/"
   	if not os.path.exists(path_out):
   		os.mkdir(path_out)
    filename = os.path.join(path_out,str(test_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)
    

df_val = pandas.read_csv("val_param.csv") 
val_params = df_val.values

for i in range(n_samp):
    omega,tau,p,D,alpha = val_params[i,1:-1]
    y = ftm.getsounds_imp(mode,mode,omega,tau,p,D,alpha,sr)
    y = y/ max(y)
    path_out = "/scratch/hh2263/drum_data/val/"
   	if not os.path.exists(path_out):
   		os.mkdir(path_out)
    filename = os.path.join(path_out,str(val_params[i,0])+"_sound.wav")
    sf.write(filename, y, sr)






