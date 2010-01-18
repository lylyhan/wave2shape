import numpy as np
import os
import soundfile as sf
from sklearn.manifold import Isomap
from sklearn.utils.graph import single_source_shortest_path_length,graph_shortest_path

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
import scipy
import gin

#metrics parameters
k = 10 #precision at k hyperparameter, can be taken maximally number of neighbors
n_nbrs = [10] #[10,50,100,500] #number of neighbors, 5% of test set
sr = 22050
eps = 0.1 #mu log coefficient
realmodels = True

feat_path = "/home/han/data/drum_data/han2022features-pkl/jtfs_j14_q12_f24/"
model_path = "/home/han/ddsp/output"
audio_path = "/home/han/data/drum_data/"
#feature hyperparameters
ftype = "cqt"
Jf = 10
Qf = 12
N = 2**16
#load jtfs feature hyperparameters
J = 14
Q = 12
num_dim = 1000
F = 24
#model parameter
nnbr = 10 #or 40, how the weightedploss is computed

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
    ranks = np.array(ranks) + 1 #ranks should start counting at 1
    return 1/N*np.sum(1/np.array(ranks)) 

def load_jtfs(fold_str, idx_reduced):
    csv_path = "../notebooks/" + fold_str + "_param_v2.csv"
    df = pd.read_csv(csv_path)
    sample_ids = df.values[:, 0]
    y = df.values[:,1:-1] #no use here
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

def find_best_trial(exp):
    min_vals = []
    for trial in os.listdir(exp):
        if os.path.isdir(os.path.join(exp,trial)):
            loss = pd.read_csv(os.path.join(exp,trial,"training.log"))
            min_vals.append(min(loss["val_loss"]))
    min_min = min(min_vals)
    return os.listdir(exp)[min_vals.index(min_min)]

def comp_jtfs(jtfs,jtfs_q1,wav,num_jtfs,num_jtfs_q1):
    #compute jtfs
    jtfs_wav = jtfs(wav)/np.linalg.norm(wav,ord=2)
    jtfs_wav_q1 = jtfs_q1(wav)/np.linalg.norm(wav,ord=2)
    #concatenation
    jtfs_comb = torch.concat((jtfs_wav_q1[:,:num_jtfs_q1,:],jtfs_wav[:,num_jtfs:,:]),dim=1)
    return jtfs_comb.squeeze()

def compute_numcoef(jtfs):
    num = 0
    wav = np.random.random(N,)
    Sw_list = jtfs(wav)
    for i in Sw_list:
        if len(i['j']) == 1:
            num += 1
    return num


if __name__ == "__main__":
    
    feature_path = "/home/han/data/drum_data/han2022features-pkl/"
   
    #dimensionality reduction 
    mean_fold, var_fold, sampvar_fold = running_var.welford(os.path.join(feature_path,                                                          "jtfs_j"+str(J)+"_q"+str(Q)+"_f"+str(F)))
    idx_fold = np.flip(np.argsort(sampvar_fold))[:num_dim]
    
    print("finished running variance computation for all")
    
    reduced_train_jtfs = load_jtfs("train",idx_fold) #slice(None) means loading full features
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
    
    
    df = pd.read_csv(os.path.join(audio_path,"annotations","test_param_v2.csv"))
    y = df.values[:,1:-1]
    sample_ids = df.values[:,0]

    #initialize jtfs instance
    params = dict(J = J, #scale
            shape = (2**16, ), 
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
    
    for n_nbr in n_nbrs:
        
        reduced_all_jtfs = []
        n_train = redlogstd_train_jtfs.shape[0]
        n_test = redlogstd_test_jtfs.shape[0]
        n_val = redlogstd_val_jtfs[0]
        redlogstd_all_jtfs = np.concatenate([redlogstd_train_jtfs,
                                    redlogstd_test_jtfs,
                                    redlogstd_val_jtfs])
            
        print("finished concatenating all features ")

        nbrs_fold = NearestNeighbors(n_neighbors=n_nbr, algorithm='ball_tree').fit(redlogstd_all_jtfs)
        print("finished fitting NN graph")
        distances_fold, indices_fold = nbrs_fold.kneighbors(redlogstd_all_jtfs)
        
        #distance matrix of NN graph
        D_nn = nbrs_fold.kneighbors_graph(redlogstd_all_jtfs, mode="distance")
        #geodesic distance matrix of NN graph
        
        #G = graph_shortest_path(dist_matrix=D_nn, directed=False)       
        print("finished computing geodesic distances")

       
        if realmodels:
            #load different experiments' predictions
            logscale = 1e-3
            loss = "MSS_cos" 
            param = "alpha"
            wt = ""
            for pm in ["gt_pitch_"]:
                if pm == "gt_pitch_":
                    pitchmode = "wopitch"
                else:
                    pitchmode = "wpitch"
                    
                exp_name = pm + loss
                model_dir = os.path.join(model_path, exp_name)
                gin_file = os.path.join(model_dir, 'operative_config-0.gin')
                
                # Parse gin config,
                with gin.unlock_config():
                    gin.parse_config_file(gin_file, skip_unknown=True)
                # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
                ckpt_files = [f for f in os.listdir(model_dir) if 'ckpt' in f]
                ckpt_name = ckpt_files[-1].split('.')[0]
                ckpt = os.path.join(model_dir, ckpt_name)
                print(ckpt)

                # Set up the model just to predict audio given new conditioning
                model = ddsp.training.models.Autoencoder()
                model.restore(ckpt)
                
                #predict
                prec_a_k = []
                ranks = []
                geo_dist = []
                for i, s_i in enumerate(sample_ids):
                    wav_path = os.path.join(audio_path, "test", str(s_i)+"_sound.wav")
                    audio,sr = sf.read(wav_path)
                    #audio_concat.append(audio)
                    #ys.append(y[i,:])
                    predicted_audio = model([tf.convert_to_tensor(audio[None,:]),
                                        tf.convert_to_tensor(y[i,:][None,:], dtype="float32")], training=False, 
                                            return_losses=True)
                    y_est = predicted_audio[0]['out']['controls']['all_est'][0,:]
                     
                    #compute jtfs
                    jtfs_wav = comp_jtfs(jtfs,jtfs_q1,y_est,num_jtfs,num_jtfs_q1)
                    #reduce dim
                    jtfs_red_wav = jtfs_wav.squeeze()[list(idx_fold)]
                    jtfs_red_wav = np.maximum(0,jtfs_red_wav.cpu().squeeze())
                    #mulog
                    jtfs_redlog_wav = np.log1p(jtfs_red_wav/(eps*mu))
                    #standardization
                    jtfs_redlogstd_wav = (jtfs_redlog_wav - mu_train) / std_train

                    #p@k
                    dist = np.linalg.norm(jtfs_redlogstd_wav - redlogstd_all_jtfs[i+n_train,:], ord=2)
                    closest_nbr = find_rank(distances_fold[i+n_train,:],dist) #distances is num_nbr long
                    rank = list(distances_fold[i+n_train,:]).index(closest_nbr)
                    ranks.append(rank) # AVEREAGE RANKING (drawback: might always be ranked last, not a fully connected graph)
                     
                    if distances_fold[i+n_train,-1] < dist: #prediction exceeds k neighborhood
                        prec_a_k.append(np.nan)
                    else:
                        prec_a_k.append(rank)


                    #query 10 closest neighbors of prediction in the graph
                    dist2nbr,clonbr_idx = nbrs_fold.kneighbors(jtfs_redlogstd_wav.reshape(1,-1),n_neighbors=n_nbr)
                    geo_nbrs = [] #neighbors of the prediction
                    dist_i = scipy.sparse.csgraph.shortest_path(csgraph=D_nn, directed=False,indices=[i+n_train])[0] #all path distances departuring from ground truth 
                    for d, ci in zip(dist2nbr,clonbr_idx):
                        #dist_nbrgeo = G[i,ci]#chosen neighbor's geodesic distance
                        dist_nbrgeo = dist_i[ci]
                        dist_geo = dist_nbrgeo + d
                        geo_nbrs.append(dist_geo)
                    geo_dist.append(np.min(geo_nbrs))


     
                np.save(loss+"_"+wt+"_from"+str(nnbr)+"nnbr_"+param+"_"+pitchmode+"_metrics_k"+str(k)+"_nbr"+str(n_nbr)+".npy",[prec_a_k, ranks, geo_dist],allow_pickle=True)













