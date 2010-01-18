"""
This script computes the nearest neighbor graph of train/test/val sets based on their JTFS coefficients. It then selects
a neighborhood of n for each sound, computes for each parameter the mean normalized parameter error of this neighborhood.
the mean parameter distance for a given neighborhood of n and graph constructed with n_nbr neighbors are then saved.
"""

import numpy as np
import os
import copy
import soundfile as sf
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
sys.path.append("../src")
import running_var
import librosa
import ftm_ver2
import sys
import math
sys.path.append("/home/han/kymatio-jtfs/")
from kymatio.scattering1d.core import timefrequency_scattering1d as tf_scat

from kymatio.torch import TimeFrequencyScattering1D,Scattering1D
import torch

feature_path = "/home/han/data/drum_data/han2022features-pkl/"
audio_path = "/home/han/data/drum_data/"

sr = 22050
N = 2**16
J = 14
Q = 12
T = 2**16 #doesnt' count here, always taking global_average=True
F = 2*Q 


k = 1000 # number of dimensions to keep at dimensionality reduction stage
n = 5 # number of neighbors to average from
n_nbr = 10 # number of neighbors for the graph
eps = 0.1 #mu log hyperparameter


def make_jtfs(jtfs,jtfs_q1,num_jtfs, num_jtfs_q1, fold_str):
    csv_path =  "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    #add the beta/alpha ground truth
    y = df.values[:,1:-1]
    ba = 1/y[:,-1] + 1/y[:,-1]**2
    y_ba = np.hstack((y, ba[:,None]))
    os.makedirs(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F),fold_str),exist_ok=True)
    jtfs_set = []
    #load sound files
    num = 0
    for i in sample_ids:
        filename = "_".join([str(i),"sound.wav"])
        file_path = os.path.join(audio_path,fold_str,filename)
        wav, sr = sf.read(file_path)
        jtfs_wav = jtfs(wav)/np.linalg.norm(wav,ord=2)
        jtfs_wav_q1 = jtfs_q1(wav)/np.linalg.norm(wav,ord=2)
        #concatenation
        jtfs_comb = torch.concat((jtfs_wav_q1[:,:num_jtfs_q1,:],jtfs_wav[:,num_jtfs:,:]),dim=1)
        jtfs_set.append(jtfs_comb.cpu())
        if not os.path.exists(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F),fold_str,
                                           fold_str+"_"+str(i)+".npy")):
            np.save(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F), fold_str,
                                 fold_str+"_"+str(i)+".npy"),jtfs_comb.cpu())
        num += 1
        if num % 100 == 0:
            print(num)
    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.reshape((jtfs_set.shape[0],
                            jtfs_set.shape[1]*jtfs_set.shape[2]*jtfs_set.shape[3])))
    return jtfs_set, y_ba

def compute_numcoef(jtfs):
    num = 0
    wav = np.random.random(N,)
    Sw_list = jtfs(wav)
    for i in Sw_list:
        if len(i['j']) == 1:
            num += 1
    return num

def load_jtfs(fold_str, idx_reduced):
    csv_path =  "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    jtfs_set = []
    num = 0
    for i in sample_ids:
        wav_path = os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F),
                                fold_str,
                                fold_str+"_"+str(i)+".npy")
        #print(wav_path)
        if os.path.exists(wav_path):
            jtfs_wav = np.load(wav_path)
            jtfs_set.append(jtfs_wav.squeeze()[idx_reduced])
            num += 1

    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.squeeze())
 
    return jtfs_set

def preprocess_gt(y_train, y_test, y_val):
    
    param_idx = [0,2,3,5]
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

    return y_train_normalized, y_test_normalized, y_val_normalized

def compute_mean_dist(indices,n,jtfs,idx_current,y_all):   
    """
    n: number of neighbors to average from
    y_current: currrent set ground truth
    y_all: ground truth parameters, all sets concatenated
    indices: (#samples, #neighbors), all sets concatenated
    
    """
    # calculate the mean parameter distance from n neighbors for each example
    param_dist = []
    eps = 1e-15 #prevent division by zero error
    start, end = idx_current
    for i in np.arange(start, end): #entire set index
        param_current = y_all[i,:].astype(np.float64) #indexed on the entire set
        #get the parameters of these neighbors 
        param_nbrs = copy.deepcopy(y_all[indices[i,:n],:]).astype(np.float64)
        #logscale w, p, D and ba
        #param_nbrs[:,[0,2,3,5]] = np.log10(param_nbrs[:,[0,2,3,5]])
        #param_current[[0,2,3,5]] = np.log10(param_current[[0,2,3,5]])
        idx_nbrs = indices[i,:n] #(k,)
        
        #take average of k neighbors to approximate jacobian matrix 
        dist = np.mean(np.abs(jtfs[idx_nbrs,:] - jtfs[i,:][None,:])[...,None] / np.abs(param_nbrs - param_current[None,:] + eps)[:,None,:], axis=0) #(k,lambda,None) / (k,None,j) -> (k,lambda,j)-> (lambda,j)
        #squaring and summing
        dist = np.sum((dist**2),axis=0) #(lambda,j) -> (j,)
        
        param_dist.append(dist)
        #param_dist.append(np.mean(np.abs(param_nbrs-param_current),axis=0)/np.abs(param_max-param_min)) #this should be the (max param - min param)

    param_dist = np.stack(param_dist) #(i,j)
    return param_dist

if __name__ == "__main__":
    
    """
    params = dict(J = J, #scale
            shape = (N, ), 
            T = N, 
            average = True,
            max_pad_factor=1,
            max_pad_factor_fr=1)
    
    # initialize jtfs (global averaging) and two octaves of frequential averaging
    jtfs = TimeFrequencyScattering1D(**params, average_fr = True, Q = Q, F = F).cuda() #same as JTFS models similarity paper
    jtfs_q1 = TimeFrequencyScattering1D(**params, average_fr = False, Q = 1).cuda()
    
    #compute number of coef
    jtfs_list = TimeFrequencyScattering1D(**params, average_fr = True, out_type = "list", Q = Q, F = F).cuda()
    num_jtfs = compute_numcoef(jtfs_list)
    jtfs_q1_list = TimeFrequencyScattering1D(**params, average_fr = False, out_type = "list", Q = 1).cuda()
    num_jtfs_q1 = compute_numcoef(jtfs_q1_list)
    
    # make JTFS features
    
    
    train_jtfs, y_train = make_jtfs(jtfs, jtfs_q1, num_jtfs, num_jtfs_q1, "train")
    test_jtfs, y_test = make_jtfs(jtfs, jtfs_q1, num_jtfs, num_jtfs_q1, "test")
    val_jtfs, y_val = make_jtfs(jtfs, jtfs_q1, num_jtfs, num_jtfs_q1, "val")
    print("finished making JTFS features")
    """
    
    y_all = []
    for fold_str in ["train","test","val"]:
        csv_path = "../notebooks/" + fold_str + "_param_v2.csv"
        df = pd.read_csv(csv_path)
        sample_ids = df.values[:, 0]
        y = df.values[:,1:-1] 
        ba = 1/y[:,-1] + 1/y[:,-1]**2
        y_ba = np.hstack((y, ba[:,None]))
        y_all.append(y_ba)
    y_train, y_test, y_val = y_all
    y_train_norm, y_test_norm, y_val_norm = preprocess_gt(y_train, y_test, y_val)
    y_all = np.concatenate([y_train_norm, y_test_norm, y_val_norm], axis=0) #normalized coefficients
    print("loaded all ground truth",y_all.shape)
    
    #dimensionality reduction based on all sets' jtfs coefficients just to derive indices
    mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,
                                                                         "jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F)))
    idx_fold = np.flip(np.argsort(sampvar_fold))[:k]
    print("finished running variance computation based on all raw data ")
    
    reduced_train_jtfs = load_jtfs("train",idx_fold) 
    reduced_test_jtfs = load_jtfs("test",idx_fold) 
    reduced_val_jtfs = load_jtfs("val",idx_fold) 
    
    #mulog
    mu = np.mean(reduced_train_jtfs,axis=0)
    redlog_train_jtfs = np.log1p(reduced_train_jtfs/(mu[None,:]*eps))
    redlog_test_jtfs = np.log1p(reduced_test_jtfs/(mu[None,:]*eps))
    redlog_val_jtfs = np.log1p(reduced_val_jtfs/(mu[None,:]*eps))
    print("finished mulog ")
    
    #standarization 
    mu_train = np.mean(redlog_train_jtfs,axis=0)
    std_train = np.std(redlog_train_jtfs,axis=0)

    redlogstd_train_jtfs = (redlog_train_jtfs - mu_train) / std_train
    redlogstd_test_jtfs = (redlog_test_jtfs - mu_train) / std_train
    redlogstd_val_jtfs = (redlog_val_jtfs - mu_train) / std_train
    print("finished standardization ")
    
    redlogstd_all_jtfs = np.concatenate([redlogstd_train_jtfs,
                                    redlogstd_test_jtfs,
                                    redlogstd_val_jtfs])
    
    nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(redlogstd_all_jtfs)
    print("finished fitting NN graph")
    distances_fold, indices_fold = nbrs_fold.kneighbors(redlogstd_all_jtfs)
    
    sets = [redlogstd_train_jtfs.shape[0], redlogstd_test_jtfs.shape[0], redlogstd_val_jtfs.shape[0]]
    start = 0
    for set_i, fold in enumerate(["train","test","val"]):
        fold_dist = compute_mean_dist(indices_fold,#[start:(start+sets[set_i]),:],
                                      n,
                                      redlogstd_all_jtfs,
                                      [start,start+sets[set_i]],
                                      y_all)
        
        with open(os.path.join(audio_path, fold + '_nbr_dist_nnbr'+str(n_nbr)+'_n'+str(n)+'jtfsavgf_nnadvanced.npy'), 'wb') as f:
            np.save(f, fold_dist)
        start = start + set_i
    
    
    """
    #training
    for fold in ["train","val","test"]:
        reduced_fold_jtfs, y_fold = load_jtfs(fold,idx_fold) #slice(None) means loading full features
        #standarization based on training set
        if fold == "train":
            #load full features       
            mu_train = np.mean(reduced_fold_jtfs,axis=0)
            std_train = np.std(reduced_fold_jtfs,axis=0)
        print("compute standardization parameters")
        
        if mu_train is not None:
            reduced_fold_jtfs = (reduced_fold_jtfs - mu_train[None,...]) / std_train[None,...]
        print("finished standardization " + fold)
       
        nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(reduced_fold_jtfs)
        distances_fold, indices_fold = nbrs_fold.kneighbors(reduced_fold_jtfs)
        print("finished building nearest neighbor graph " + fold)
        # compute distance
        fold_dist = compute_mean_dist(indices_fold,n,y_fold)
        print("finished computing distance " + fold)
        
        #nnbr is number of neighbors used when computing nearest neighbor graph, n is number of neighbors to take the mean from
        with open(os.path.join(audio_path, fold + '_nbr_dist_nnbr'+str(n_nbr)+'_n'+str(n)+'jtfsavgf.npy'), 'wb') as f:
            np.save(f, fold_dist)
            
        del reduced_fold_jtfs,y_fold
    
    """

    
    
    
    
