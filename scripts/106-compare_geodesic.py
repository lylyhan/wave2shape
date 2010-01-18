"""
This script will pick a reference point in the test set, compute a series of pitch perturbed sounds,
their geodesic distance to the referenc point, and save the distances into a npy file.
"""

import numpy as np
import os
import soundfile as sf
from sklearn.manifold import Isomap
import sys
import tensorflow as tf
sys.path.append("../src")
sys.path.append("/home/han/kymatio-jtfs/")
sys.path.append("../scripts")

import train
import features
import data_generator
import cnn
import ftm_ver2
import running_var

import pandas as pd
import librosa
import IPython.display as ipd
from kymatio.torch import TimeFrequencyScattering1D,Scattering1D
import torch
from kymatio.scattering1d.core import timefrequency_scattering1d as tf_scat
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import tensorflow.keras.backend as K


n_nbrs = [10, 50, 100, 500]
sr = 22050
eps = 0.1 #mu log coefficient

feat_path = "/home/han/data/drum_data/han2022features-pkl/jtfs_j14_q16/"
model_path = "/home/han/wave2shape/output/doce"
audio_path = "/home/han/data/drum_data/"


#jtfs hyperparameters
J = 14
Q = 12
F = 24
N = 2**16

def load_jtfs(fold_str, idx_reduced):
    csv_path = "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    y = df.values[:,1:-1]
    ba = 1/y[:,-1] + 1/y[:,-1]**2
    y_ba = np.hstack((y, ba[:,None]))
    
    jtfs_set = []
    num = 0
    for i in sample_ids:
        wav_path = os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F),
                                fold_str,
                                fold_str+"_"+str(i)+".npy")
        if os.path.exists(wav_path):
            jtfs_wav = np.load(wav_path)
            jtfs_set.append(jtfs_wav.squeeze()[idx_reduced])
            num += 1

    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.squeeze())
    #mulogscale jtfs
    if fold_str == "test":
        #mulog
        mu = np.mean(jtfs_set,axis=0)
        jtfs_set = np.log1p(jtfs_set/eps/mu[None,:])
        return jtfs_set,y_ba,mu
    else:
        mu = np.mean(jtfs_set,axis=0)
        jtfs_set = np.log1p(jtfs_set/eps/mu[None,:])
        return jtfs_set,y_ba
    
def compute_numcoef(jtfs):
    num = 0
    wav = np.random.random(N,)
    Sw_list = jtfs(wav)
    for i in Sw_list:
        if len(i['j']) == 1:
            num += 1
    return num

if __name__ == "__main__":
    
    #compute nearest neighbor graph of jtfs features of the entire test set
    fold = "test"
    feature_path = "/home/han/data/drum_data/han2022features-pkl/"
    num_dim = 1000
 

    mu_train = None
    std_train = None
    
    #dimensionality reduction based on all sets' jtfs coefficients just to derive indices
    mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,
                                                                         "jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F)))
    idx_fold = np.flip(np.argsort(sampvar_fold))[:num_dim]
    print("finished running variance computation based on all raw data ")
    
    reduced_train_jtfs, y_fold = load_jtfs("train",idx_fold)
    mu_train = np.mean(reduced_train_jtfs,axis=0)
    std_train = np.std(reduced_train_jtfs,axis=0)
  
    print("compute standardization parameters")

    
    reduced_fold_jtfs, y_fold, mu_fold = load_jtfs(fold,idx_fold)
    reduced_fold_jtfs = (reduced_fold_jtfs - mu_train[None,...]) / std_train[None,...]
    print("finished loading features with reduced dimensionality and standardize " + fold)
    
    #initialize jtfs instance
    params = dict(J = J, #scale
            shape = (N, ), 
            T = N, 
            average = True,
            max_pad_factor=1,
            max_pad_factor_fr=1)
    
    ## when frequential averaging is true, make jtfs this way:
    # initialize jtfs (global averaging) and two octaves of frequential averaging
    jtfs = TimeFrequencyScattering1D(**params, average_fr = True, Q = Q, F = F).cuda() #same as JTFS models similarity paper
    jtfs_q1 = TimeFrequencyScattering1D(**params, average_fr = False, Q = 1).cuda()
    
    #compute number of coef
    jtfs_list = TimeFrequencyScattering1D(**params, average_fr = True, out_type = "list", Q = Q, F = F).cuda()
    num_jtfs = compute_numcoef(jtfs_list)
    jtfs_q1_list = TimeFrequencyScattering1D(**params, average_fr = False, out_type = "list", Q = 1).cuda()
    num_jtfs_q1 = compute_numcoef(jtfs_q1_list)
    
  
    
   
    geo_dist = {10:[],50:[],100:[],500:[],"euc":[]}
    for n_nbr in n_nbrs:
        nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(reduced_fold_jtfs)
        distances_fold, indices_fold = nbrs_fold.kneighbors(reduced_fold_jtfs)
        print("finished building nearest neighbor graph for nnbr" + str(n_nbr))

        #compute isomap
        isomap_fold = Isomap(n_neighbors=n_nbr).fit(reduced_fold_jtfs)

        #pick a random reference in test set
        y_test,test_ids = train.load_gt("test", param="alpha")
        i = 500 # random index
        omega,tau,p,D,alpha = y_test[i,:]
        for d_o in np.logspace(np.log10(0.5),np.log10(2),200): #how much to perturb the pitch
            omega_pert = omega*d_o
            #synthesize perturbed audio
            wavform = ftm_ver2.getsounds_dif_linear_nonorm(10,10,0.4,0.4,0.03,tau,
                                                        omega_pert,p,D,np.pi,alpha,sr)
            wavform = wavform/ max(wavform)

            #compute jtfs
            jtfs_wav = jtfs(wavform)/np.linalg.norm(wavform,ord=2)
            jtfs_wav_q1 = jtfs_q1(wavform)/np.linalg.norm(wavform,ord=2)
            #concatenation
            jtfs_comb = torch.concat((jtfs_wav_q1[:,:num_jtfs_q1,:],jtfs_wav[:,num_jtfs:,:]),dim=1)
            
            #apply dimensionality reduction
            jtfs_wav_red = jtfs_comb.squeeze()[list(idx_fold)]
            jtfs_wav_red = np.maximum(0,jtfs_wav_red.cpu().squeeze())
            #mulog
            jtfs_wav_red = np.log1p(jtfs_wav_red/eps/mu_fold)
            #standardizee
            jtfs_wav_red = (jtfs_wav_red - mu_train) / std_train
            
            #compute naive euclidean distance
            dist = np.linalg.norm(jtfs_wav_red - reduced_fold_jtfs[i,:], ord=2)
            if n_nbr == 10:
                geo_dist["euc"].append((omega_pert, dist))
            
            #compute euclidean distance to the chosen neighbor 
            #query the closest neighbor of this prediction
            dist2nbr,clonbr_idx = nbrs_fold.kneighbors(jtfs_wav_red.reshape(1,-1),n_neighbors=1)
            #extract geodesic distance of this neighbor to groundtruth
            dist_nbrgeo = isomap_fold.dist_matrix_[i,clonbr_idx]#chosen neighbor's geodesic distance
            dist_geo = dist_nbrgeo + dist2nbr
            geo_dist[n_nbr].append((omega_pert, dist_geo))
    np.save("geodesic_varynnbr_jtfsfavg.npy", geo_dist, allow_pickle=True)



            



