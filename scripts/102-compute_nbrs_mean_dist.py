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
from kymatio.scattering1d.core import timefrequency_scattering1d as tf_scat

from kymatio.torch import TimeFrequencyScattering1D,Scattering1D
import torch

feature_path = "/home/han/data/drum_data/han2022features-pkl/"
audio_path = "/home/han/data/drum_data/"

sr = 22050
N = 2**16
J = 14
Q = 16
T = 2**11 #doesnt' count here, always taking global_average=True

k = 1000 # number of dimensions to keep at dimensionality reduction stage
n = 15 # number of neighbors to average from
n_nbr = 40 # number of neighbors for the graph

def make_jtfs(jtfs,fold_str):
    csv_path =  "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    #add the beta/alpha ground truth
    y = df.values[:,1:-1]
    ba = 1/y[:,-1] + 1/y[:,-1]**2
    y_ba = np.hstack((y, ba[:,None]))
    os.makedirs(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q)),exist_ok=True)
    jtfs_set = []
    #load sound files
    num = 0
    for i in sample_ids:
        filename = "_".join([str(i),"sound.wav"])
        file_path = os.path.join(audio_path,fold_str,filename)
        wav, sr = sf.read(file_path)
        jtfs_wav = jtfs(wav)/np.linalg.norm(wav,ord=2)
        jtfs_set.append(jtfs_wav.cpu())
        if not os.path.exists(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q),
                                           fold_str+"_"+str(i)+".npy")):
            np.save(os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q),
                                 fold_str+"_"+str(i)+".npy"),jtfs_wav.cpu())
        num += 1
        if num % 100 == 0:
            print(num)
    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.reshape((jtfs_set.shape[0],
                            jtfs_set.shape[1]*jtfs_set.shape[2]*jtfs_set.shape[3])))
    return jtfs_set, y_ba

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
        wav_path = os.path.join(feature_path,"jtfs_j"+str(J)+"_q"+str(Q),fold_str,
                                       fold_str+"_"+str(i)+".npy")
        if os.path.exists(wav_path):
            jtfs_wav = np.load(wav_path)
            jtfs_set.append(jtfs_wav.squeeze()[idx_reduced])
            num += 1
        if num % 100 == 0:
            print(num)
    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.squeeze())
    #jtfs_set = np.maximum(0,jtfs_set.reshape((jtfs_set.shape[0],
    #                        jtfs_set.shape[1]*jtfs_set.shape[2]*jtfs_set.shape[3])))
    return jtfs_set,y_ba


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

if __name__ == "__main__":

    
    # initialize jtfs (global averaging)
    #jtfs = TimeFrequencyScattering1D(
    #                J = J, #scale
    #                shape = (N, ), 
    #                Q = Q, #filters per octave, frequency resolution
    #                T = N, 
    #                average = True,
    #                max_pad_factor=1,
    #                max_pad_factor_fr=1,
    #                average_fr = False,
    #            ).cuda()
    
    # make JTFS features
    
    #train_jtfs, y_train = make_jtfs(jtfs,"train")
    #test_jtfs, y_test = make_jtfs(jtfs,"test")
    #val_jtfs, y_val = make_jtfs(jtfs,"val")
    #print("finished making JTFS features")
    
    
    #training
    for fold in ["test","train"]:
        mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,
                                                                             "jtfs_j"+str(J)+"_q"+str(Q),
                                                                             fold))
        print("finished running variance computation " + fold)
        idx_fold = np.flip(np.argsort(sampvar_fold))[:k]
        reduced_fold_jtfs, y_fold = load_jtfs(fold,idx_fold)
        print("finished loading features with reduced dimensionality" + fold)
       # reduced_fold_jtfs = fold_jtfs[:,idx_fold]
        #print("finished reducing dimensionality " + fold)
        nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(reduced_fold_jtfs)
        distances_fold, indices_fold = nbrs_fold.kneighbors(reduced_fold_jtfs)
        print("finished building nearest neighbor graph " + fold)
        # compute distance
        fold_dist = compute_mean_dist(indices_fold,n,y_fold)
        print("finished computing distance " + fold)
        with open(os.path.join(audio_path, fold + '_nbr_dist.npy'), 'wb') as f:
            np.save(f, fold_dist)
            
        del reduced_fold_jtfs,y_fold
    
    

    
    
    
    
