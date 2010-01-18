"""
This script computes the nearest neighbor graph of train/test/val sets based on MSS distances. It then selects
a neighborhood of n for each sound, computes for each parameter the mean normalized parameter error of this neighborhood.
the mean parameter distance for a given neighborhood of n and graph constructed with n_nbr neighbors are then saved.
"""

import numpy as np
import os
import copy
import soundfile as sf
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import sys
sys.path.append("../src")
import running_var
import librosa
import ftm_ver2
import sys
sys.path.append("/home/han/kymatio-jtfs/")
import ddsp
from ddsp.training import models, eval_util, ddsp_run
import torch
import tensorflow as tf


feature_path = "/home/han/data/drum_data/han2022features-pkl/"
audio_path = "/home/han/data/drum_data/"

sr = 22050
N = 2**16

n = 5 # number of neighbors to average from
n_nbr = 10 # number of neighbors for the graph
k = 1000 #dim reduction 
eps = 0.1

def make_MSS(fold_str,sploss):
    csv_path =  "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    
    os.makedirs(os.path.join(feature_path,"MSS_tavged",fold_str),exist_ok=True)
    num = 0
    mss_set = []
    for i in sample_ids:
        filename = "_".join([str(i),"sound.wav"])
        file_path = os.path.join(audio_path,fold_str,filename)
        wav, sr = sf.read(file_path)
        mags = []
        for loss_op in sploss.spectrogram_ops:
            mag = np.sum(loss_op(wav).numpy(), axis=1) #equivalent to average global
            mags.append(mag)
        mags = np.concatenate(mags)
        mss_set.append(mags)
        if not os.path.exists(os.path.join(feature_path,"MSS_tavged",fold_str,
                                           fold_str+"_"+str(i)+".npy")):
            np.save(os.path.join(feature_path,"MSS_tavged", fold_str,
                                 fold_str+"_"+str(i)+".npy"),mags)
        num += 1
        
        if num % 100 == 0:
            print(num)
    mss_set = np.stack(mss_set)
    mss_set = np.maximum(0,mss_set.reshape((mss_set.shape[0], np.product(mss_set.shape[1::]))))
    return mss_set
    
def load_mss(fold_str, idx_reduced):
    csv_path =  "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    mss_set = []
    num = 0
    for i in sample_ids:
        wav_path = os.path.join(feature_path,"MSS_tavged",
                                fold_str,
                                fold_str+"_"+str(i)+".npy")
        #print(wav_path)
        if os.path.exists(wav_path):
            mss_wav = np.load(wav_path)
            mss_set.append(mss_wav.squeeze()[idx_reduced])
            num += 1

    mss_set = np.stack(mss_set)
    mss_set = np.maximum(0, mss_set.squeeze())
 
    return mss_set
                         
    

def compute_mean_dist(indices,n,y):   
    """
    n: number of neighbors to average from
    y: ground truth parameters
    """
    # calculate the mean parameter distance from n neighbors for each example
    param_dist = []
    for i in range(indices.shape[0]):
        param_current = y[i,:].astype(np.float64)
        #get the parameters of these neighbors 
        param_nbrs = copy.deepcopy(y[indices[i,:n],:]).astype(np.float64)
        #logscale p and D and ba
        param_nbrs[:,[0,2,3,5]] = np.log10(param_nbrs[:,[0,2,3,5]])
        param_current[[0,2,3,5]] = np.log10(param_current[[0,2,3,5]])
        param_dist.append(np.mean(np.abs(param_nbrs-param_current),axis=0)/np.abs(param_current))

    param_dist = np.stack(param_dist)
    return param_dist

def sum_series(n_terms,max_start,gap):
    return int((max_start + (max_start + (n_terms-1) *gap)) * n_terms / 2)
    
def pair2idx(i,j):
    i,j = sorted([i, j]) #sort indices from small to big
    start = sum_series(i, 100000, -1) #start idx correponding to i
    return start + (j - i)

def batch_mss(target_mags, batch, sploss):
    #assert batch1.shape == batch2.shape
    assert len(batch.shape) == 2
    
    #target_mags = []
    #for loss_op in sploss.spectrogram_ops:
    #    target_mags.append(loss_op(target))
        
    loss = np.zeros((batch.shape[0],)) 
    for i, loss_op in enumerate(sploss.spectrogram_ops):
        #target_mag = loss_op(batch1)
        batch_mag = loss_op(batch)
        difference = target_mags[i][None,...] - batch_mag
        weights = 1
        loss += tf.reduce_mean(tf.abs(difference * weights),axis=[1,2])
    return loss

if __name__ == "__main__":
    #define MSS distance
    sploss = ddsp.losses.SpectralLoss()
    
    
    #make time-averaged MSS
    #train_MSS = make_MSS("train", sploss)
    #test_MSS = make_MSS("test", sploss)
    #val_MSS = make_MSS("val", sploss)
    print("finished making all time-averaged MSS features")
    
    #load all ground truth and sounds
    y_all = []
    sets = []
    fnames = []
    for fold_str in ["train","test","val"]:
        csv_path = "../notebooks/" + fold_str + "_param_v2.csv"
        df = pd.read_csv(csv_path)
        sample_ids = df.values[:, 0]
        y = df.values[:,1:-1] 
        for s_id in sample_ids:
            fnames.append(fold_str + "/" + str(s_id) + "_sound.wav")
        sets.append(y.shape[0])
        ba = 1/y[:,-1] + 1/y[:,-1]**2
        y_ba = np.hstack((y, ba[:,None]))
        y_all.append(y_ba)
    y_all = np.concatenate(y_all, axis=0)

    print("loaded all ground truth",y_all.shape)
    
     #dimensionality reduction based on all sets' mss coefficients just to derive indices
    mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,
                                                                         "MSS_tavged"))
    idx_fold = np.flip(np.argsort(sampvar_fold))[:k]
    print("finished running variance computation based on all raw data ")
    
    reduced_train_mss = load_mss("train",idx_fold) 
    reduced_test_mss = load_mss("test",idx_fold) 
    reduced_val_mss = load_mss("val",idx_fold) 
    
    #mulog
    mu = np.mean(reduced_train_mss,axis=0)
    redlog_train_mss = np.log1p(reduced_train_mss/(mu[None,:]*eps))
    redlog_test_mss = np.log1p(reduced_test_mss/(mu[None,:]*eps))
    redlog_val_mss = np.log1p(reduced_val_mss/(mu[None,:]*eps))
    print("finished mulog ")
    
    #standarization 
    mu_train = np.mean(redlog_train_mss,axis=0)
    std_train = np.std(redlog_train_mss,axis=0)

    redlogstd_train_mss = (redlog_train_mss - mu_train) / std_train
    redlogstd_test_mss = (redlog_test_mss - mu_train) / std_train
    redlogstd_val_mss = (redlog_val_mss - mu_train) / std_train
    print("finished standardization ")
    
    redlogstd_all_mss = np.concatenate([redlogstd_train_mss,
                                    redlogstd_test_mss,
                                    redlogstd_val_mss])
    
    nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(redlogstd_all_mss)
    print("finished fitting NN graph")
    distances_fold, indices_fold = nbrs_fold.kneighbors(redlogstd_all_mss)
    
    
    """
    #store distances
    dists = np.zeros((int(1+sum(sets)*sum(sets)/2),))
    start = 0
    for i in range(sum(sets)):
        wav_path = os.path.join(audio_path, fnames[i])
        wav, sr = sf.read(wav_path)
        
        #compute MSS
        target_mags = []
        for loss_op in sploss.spectrogram_ops:
            target_mags.append(loss_op(wav))

        n = 1000
        for n_j in range(sum(sets) // n + 1): #partition all audios to some parts
            end = min(n_j * n + n, sum(sets))
            start = n_j * n
            print("progress: ",i,start, end)
            wavs = []
            for j in range(start, end):
                wav_path2 = os.path.join(audio_path, fnames[i])
                wav2, sr = sf.read(wav_path2)
                wavs.append(wav2)
            print("batchsize",len(wavs))
            if len(wavs) == 0:
                pass
            else:
                loss = batch_mss(target_mags, np.stack(wavs, axis=0), sploss) #(n,)
                for j in range(start, end):
                    dists[pair2idx(i,j)] = loss[j-start]
            
    print("finished computing distances!")
    
    fake_data = np.arange(sum(sets))
    nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='auto',
                                metric=lambda a,b: dists[pair2idx(a,b)]).fit(fake_data)
    print("finished fitting NN graph")
    distances_fold, indices_fold = nbrs_fold.kneighbors(fake_data)
    
    """
    
    
    start = 0
    for set_i, fold in enumerate(["train","val","test"]):
        fold_dist = compute_mean_dist(indices_fold[start:(start+sets[set_i]),:],
                                      n,
                                      y_all)
        
        with open(os.path.join(audio_path, fold + '_nbr_dist_nnbr'+str(n_nbr)+'_n'+str(n)+'mss.npy'), 'wb') as f:
            np.save(f, fold_dist)
        start = start + set_i
    
    
    
    
    
    
