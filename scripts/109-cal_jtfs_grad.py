"""
This script calculates the gradient(averaged over path) of JTFS coefficient with respect to each normalized parameter
and save them according to dataset. there will be a vector of 5 associated to each sample.
"""
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
import sys
sys.path.append("../src")
sys.path.append("/home/han/kymatio-jtfs/")
sys.path.append("../scripts")
import ftm_ver2 as ftm
import pandas as pd
import librosa
import IPython.display as ipd
from kymatio.torch import TimeFrequencyScattering1D,Scattering1D
import torch
from kymatio.scattering1d.core import timefrequency_scattering1d as tf_scat
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import copy
import math
import functorch

csv_path = "/home/han/data/drum_data/annotations/"
#FTM in torch
n_samples = 2**16
sample_rate = 22050

def getsounds_imp_linear_nonorm(m1,m2,x1,x2,h,theta,l0):
    """
    This implements Rabenstein's drum model. The inverse SLT operation is done at the end of each second-
    -order filter, no normalization on length and side length ratio is done
    note that batch calculation is not allowed since each sound might result in different max allowed mode numbers
    """
    #print(w11,tau11,p,D,l0,alpha_side)
    w11 = theta[:,0]
    tau11 = theta[:,1]
    p = theta[:,2]
    D = theta[:,3]
    l0 = torch.tensor(l0)
    alpha_side = theta[:,4]

    l2 = l0 * alpha_side 
    s11 = -1 / tau11
    pi = torch.tensor(np.pi)

    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2 + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)) * (l0 / np.pi)**2
    T = c_sq 
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi)**2 

    EI = S**4 

    mu = torch.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = torch.arange(1,m2+1)
    #print(mu.shape,mu2.shape,l2.shape,mu,mu2)
    dur = n_samples
    Ts = 1/sample_rate
    n = (mu[None,:] * pi / l0)**2 + (mu2[None,:] * pi / l2[:,None])**2 #eta 
    n2 = n**2 
    K = torch.sin(mu * pi * x1) * torch.sin(mu2 * pi * x2) #mu pi x / l (mode)
    #print(n.shape,n2.shape,K.shape)

    beta = EI[:,None] * n2[None,:] + T[:,None] * n[None,:] #(1,bs,m)
    alpha = (d1[:,None] - d3[:,None] * n[None,:])/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))
    #print(beta.shape,alpha.shape,omega.shape)
    #insert adaptively change mode number
    temp = (omega/2/pi) <= sample_rate / 2
    mode_corr = torch.sum(temp.to(torch.int32),) #each sample in the batch has its own mode_corr
    
    N = l0 * l2 / 4
    yi = h * torch.sin(mu[None,None,:mode_corr] * pi * x1) * torch.sin(mu2[None,None,:mode_corr] * pi * x2) / omega[:,:,:mode_corr] #(1,bs,mode)

    time_steps = torch.linspace(0,dur,dur) / sample_rate #(T,)
    y = torch.exp(-alpha[:,:,:mode_corr,None] * time_steps[None,None,None,:]) * torch.sin(omega[:,:,:mode_corr,None] * time_steps[None,None,None,:]) # (1,bs,mode,T)

    y = yi[...,None] * y

    y = torch.sum(y * K[None,None,:mode_corr,None] / N[None,:,None,None],axis=-2) #impulse response itself
    y = y / torch.max(y,dim=-1).values[...,None]
    #print(y.shape)
    return y

def preprocess_gt(y_train, y_test, y_val):
    
    param_idx = [0,2,3]
    y_train_cp = copy.deepcopy(y_train)
    y_test_cp = copy.deepcopy(y_test)
    y_val_cp = copy.deepcopy(y_val)
    
    #logscale
    for idx in param_idx:
        y_train_cp[:,idx] = [math.log10(i) for i in y_train_cp[:,idx]]
        y_test_cp[:,idx] = [math.log10(i) for i in y_test_cp[:,idx]]
        y_val_cp[:,idx] = [math.log10(i) for i in y_val_cp[:,idx]]
        
    #normalize
    scaler = MinMaxScaler()
    scaler.fit(y_train_cp)
    y_train_normalized = scaler.transform(y_train_cp)
    y_val_normalized = scaler.transform(y_val_cp)
    y_test_normalized = scaler.transform(y_test_cp)

    return y_train_normalized, y_test_normalized, y_val_normalized, scaler

def inverse_scale(y_norm,scaler):
    sc_max = torch.tensor(scaler.data_max_)
    sc_min = torch.tensor(scaler.data_min_)
    
    param_idx = [0,2,3]
    y_norm_o = y_norm * (sc_max - sc_min) + sc_min
    helper = torch.ones(y_norm_o.shape)
    #inverse logscale
    for idx in param_idx:
        helper[idx] = torch.pow(10,y_norm_o[idx]) / y_norm_o[idx]
    y_norm_o = y_norm_o * helper
    return y_norm_o



if __name__ == "__main__":
    
    #load original parameters and normalize
    df_train = pd.read_csv(os.path.join(csv_path, "train_param_v2.csv"))
    df_test = pd.read_csv(os.path.join(csv_path, "test_param_v2.csv"))
    df_val = pd.read_csv(os.path.join(csv_path, "val_param_v2.csv"))
    y_train = df_train.values[:,1:-1].astype(np.float64)
    y_test = df_test.values[:,1:-1].astype(np.float64)
    y_val = df_val.values[:,1:-1].astype(np.float64)
    y_train_norm, y_test_norm, y_val_norm, scaler = preprocess_gt(y_train, y_test, y_val)
    
    
    jtfs = TimeFrequencyScattering1D(
            J = 14, #scale
            shape = (2**16, ), 
            Q = 1, #filters per octave, frequency resolution
            T = 2**16, 
            F = 2,
            max_pad_factor=1,
            max_pad_factor_fr=1,
            average = True,
            average_fr = True,
        ).cuda()
    
    def cal_jtfs(param_n):
        param_o = inverse_scale(param_n, scaler) 
        wav1 = getsounds_imp_linear_nonorm(m1,m2,x1,x2,h,param_o[None,:],l0)
        jwav = jtfs(wav1).squeeze()
        return jwav

    m1 = m2 = 10
    x1 = x2 = 0.4
    h = 0.03
    l0 = np.pi
    batchsize = 10
    sets = ["train", "test", "val"]
    for j, param_norm in enumerate([y_train_norm, y_test_norm, y_val_norm]):
        print("making gradients for set ", sets[j])
        set_grad = []
       
        for i in range(param_norm.shape[0]): #param_norm.shape[0]): #iterate over each sample in the dataset
            if i%1000 == 0:
                print(i)
            #scale normalized param back to original ranges
            torch.autograd.set_detect_anomaly(True)
            param_n = torch.tensor(param_norm[i,:], requires_grad=True) #where the gradient starts taping
            #print(param_n.shape)

            grads = functorch.jacfwd(cal_jtfs)(param_n) #(639,5)
            JTJ = torch.matmul(grads.T, grads)
            #avg_grads = torch.mean(grads**2, axis=0)
            #print(avg_grads)
            #print(JTJ.shape)
            set_grad.append(JTJ.cpu().detach().numpy())
            torch.cuda.empty_cache()
        set_grad = np.stack(set_grad, axis=0)    
        

        np.save("/home/han/data/" + sets[j] + "_grad_jtfs.npy",set_grad)
