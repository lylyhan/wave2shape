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
import pescador
from tqdm import tqdm
import tensorflow.keras.backend as K


k = 10 #precision at k hyperparameter, can be taken maximally number of neighbors
n_nbr = 10 #number of neighbors, 5% of test set
sr = 44100
eps = 0.1 #mu log coefficient

feat_path = "/home/han/data/drum_data/han2022features-pkl/jtfs_j14_q16/"
model_path = "/home/han/wave2shape/output/doce"
audio_path = "/home/han/data/drum_data/"


def inverse_scale(y_predicted,scaler,param):
    if param == "beta_alpha":
        param_idx = [0,2,3,4]
    else:
        param_idx = [0,2,3]
    y_predicted_o = scaler.inverse_transform(y_predicted)
    #inverse logscale
    for idx in param_idx:
        y_predicted_o[:,idx] = [pow(10,i) for i in y_predicted_o[:,idx]]
    return y_predicted_o

def find_rank(a,b):
    n = len(a)
    if n == 1:
        return a[0] #rank of a 
    else:
        if b > a[n//2]:
            return find_rank(a[n//2:],b)
        else:
            return find_rank(a[:n//2],b)
        
def mean_recipro_rank(ranks): #closer to 1 the higher ranking
    N = len(ranks)
    return 1/N*np.sum(1/np.array(ranks))

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

    jtfs_set = np.stack(jtfs_set)
    jtfs_set = np.maximum(0,jtfs_set.squeeze())
    #mulogscale jtfs
    mu = np.mean(jtfs_set,axis=0)
    jtfs_set = np.log1p(jtfs_set/eps/mu[None,:])
    
    return jtfs_set,y_ba,mu

def find_best_trial(exp):
    min_vals = []
    for trial in os.listdir(exp):
        if os.path.isdir(os.path.join(exp,trial)):
            loss = pd.read_csv(os.path.join(exp,trial,"training.log"))
            min_vals.append(min(loss["val_loss"]))
    min_min = min(min_vals)
    return os.listdir(exp)[min_vals.index(min_min)]





if __name__ == "__main__":

    
    #compute nearest neighbor graph of jtfs features of the entire test set
    fold = "test"
    
    feature_path = "/home/han/data/drum_data/han2022features-pkl/"
    J = 14
    Q = 16
    num_dim = 1000
 
    
    mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,
                                                                         "jtfs_j"+str(J)+"_q"+str(Q),
                                                                         fold))
    print("finished running variance computation " + fold)
    idx_fold = np.flip(np.argsort(sampvar_fold))[:num_dim]
    reduced_fold_jtfs, y_fold, mu_fold = load_jtfs(fold,idx_fold)
    print("finished loading features with reduced dimensionality " + fold)
    # reduced_fold_jtfs = fold_jtfs[:,idx_fold]
    #print("finished reducing dimensionality " + fold)
    nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(reduced_fold_jtfs)
    distances_fold, indices_fold = nbrs_fold.kneighbors(reduced_fold_jtfs)
    print("finished building nearest neighbor graph " + fold)
    
    #compute isomap
    isomap_fold = Isomap(n_neighbors=n_nbr).fit(reduced_fold_jtfs)
    
    jtfs = TimeFrequencyScattering1D(
                J = 14, #scale
                shape = (2**16, ), 
                Q = 16, #filters per octave, frequency resolution
                T = 2**16, 
                max_pad_factor=1,
                max_pad_factor_fr=1,
                average = True,
                average_fr = False,
            ).cuda()
    
    #load different experiments' predictions
    ftype = "cqt"
    J = 8
    Q = 16
    logscale = 1e-3
    
    y_preds = {"ploss":{},"weighted_p":{}}
    for loss in ["ploss","weighted_p"]:
        for param in ["alpha", "beta_alpha"]:
            exp_type="_".join(["multitask"+str(True), loss, param, "linear", "log"+str(logscale)])
            exp_name = "_".join([ftype,"J"+str(J),"Q"+str(Q),"bs"+str(64),exp_type])
            trial_name = find_best_trial(os.path.join(model_path,exp_name))
            print(exp_name,trial_name)
            
            #redo scaling procedures to obtain scaler
            y_test,test_ids = train.load_gt("test", param)
            y_train,train_ids = train.load_gt("train", param)
            y_val,val_ids = train.load_gt("val", param)
            y_train_normalized, y_val_normalized, y_test_normalized, scaler = train.preprocess_gt(y_train,y_val,y_test,param) 


            # load predictions
            y_pred,y_gt = np.load(os.path.join(model_path,exp_name,trial_name,"test_preds.npy"))        
            y_preds[loss][param] = y_pred
            
            #inverse scale the predictions
            y_predicted_o = inverse_scale(y_pred,scaler,param)
            
            #resynthesize predictions, compute jtfs
            prec_a_k = []
            ranks = []
            geo_dist = []
            for i in range(y_predicted_o.shape[0]):
                #synthesize audio
                omega,tau,p,D,ba = y_predicted_o[i,:]
                root = max(np.roots([ba,-1,-1]))
                alpha = min(root,1/root) # could be a problem when alpha too small
    
                wavform = ftm_ver2.getsounds_dif_linear_nonorm(10,10,0.5,0.5,0.03,tau,
                                                    omega,p,D,np.pi,alpha,sr)
                wavform = wavform/ max(wavform)
            
                #compute jtfs
                jtfs_wav = jtfs(np.array(wavform))/np.linalg.norm(wavform,ord=2)
                
                #apply dimensionality reduction
                jtfs_wav_red = jtfs_wav.squeeze()[list(idx_fold)]
                jtfs_wav_red = np.maximum(0,jtfs_wav_red.cpu().squeeze())
                #mulog
                jtfs_wav_red = np.log1p(jtfs_wav_red/eps/mu_fold)
                
                #measure euclidean distance to reference and compare with nth neighbor distance
                dist = np.linalg.norm(jtfs_wav_red - reduced_fold_jtfs[i,:], ord=2)
                closest_nbr = find_rank(distances_fold[i,:],dist) #distances is num_nbr long
                rank = list(distances_fold[i,:]).index(closest_nbr)
                ranks.append(rank) # AVEREAGE RANKING (drawback: might always be ranked last, not a fully connected graph)
                
                if distances_fold[i,k-1] < dist: #prediction exceeds k neighborhood
                    prec_a_k.append(np.nan)
                else:
                    prec_a_k.append(rank)
                    
                #compute euclidean distance to the chosen neighbor 
                #query the closest neighbor of this prediction
                dist2nbr,clonbr_idx = nbrs_fold.kneighbors(jtfs_wav_red.reshape(1,-1),n_neighbors=1)
                #extract geodesic distance of this neighbor to groundtruth
                dist_nbrgeo = isomap_fold.dist_matrix_[i,clonbr_idx]#chosen neighbor's geodesic distance
                dist_geo = dist_nbrgeo + dist2nbr
                geo_dist.append(dist_geo)
            
            print(loss,param," average rank, average geoedesic distance,",mean_recipro_rank(ranks),np.mean(np.array(geo_dist)))
            np.save(loss+"_"+param+"metrics_k"+str(k)+"_nbr"+str(n_nbr)+".npy",[prec_a_k, ranks, geo_dist],allow_pickle=True)
            

                   

            



            
