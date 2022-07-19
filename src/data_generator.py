import librosa
import pescador
import features
import soundfile as sf
import numpy as np
import random
import os

audio_path = "/home/han/data/drum_data/"


@pescador.streamable
def feature_sampler(ids, fold, params_normalized, fold_dist, idx, audio_path, J, Q, ftype, loss, param, eps):
    """
    output a {input, ground truth} pair for the designated audio sample
    """
    i = idx
    y = params_normalized[i,:] #ground truth
    #load weights here!
    if fold_dist is None:
        weights = np.ones((5,)) #does not affect the following computation
    else:
        #if param == "beta_alpha":
        #    param_idx = [0,1,2,3,5]
        #else:
        #    param_idx = [0,1,2,3,4]
        weights = np.log1p(1/fold_dist[i,:])
    
    fullpath = os.path.join(audio_path,fold,str(ids[i])+"_sound.wav") 
    x,sr = sf.read(fullpath)
    fmin = 0.4*sr/np.power(2,J)
    if ftype == "cqt":
        Sy = features.make_cqt(x,b=Q,sr=sr,n_oct=J,fmin=fmin)
        nfreq,ntime = Sy.shape
        Sy = Sy[:,:,None]
    elif ftype == "vqt":
        Sy = features.make_vqt(x,b=Q,sr=sr,n_oct=J,fmin=fmin)
        nfreq,ntime = Sy.shape
        Sy = Sy[:,:,None]
    elif ftype == "hcqt":
        Sy = features.make_hcqt(x,b=Q,sr=sr,n_oct=J-2,fmin=fmin)
        nharm,nfreq,ntime = Sy.shape
        Sy = Sy.reshape((nfreq,ntime,nharm))
    #logscale the input
    if eps:
        Sy = np.log1p(Sy/eps)
    #print(Sy.shape)
    if loss == "weighted_p":
        while True:
            yield {'input': Sy,'y': y,"weights":weights[:,None]} #third key is supposed to be the sample weight!!
    else:
        while True:
            yield {'input': Sy,'y': y}

def data_generator(ids,fold,params_normalized,feature_type,audio_path, batch_size, idx, active_streamers,
                        rate,loss, param, eps,random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    #load the weights npy file
    fold_dist = np.load(os.path.join(audio_path, fold + "_nbr_dist.npy"))
    streams = [feature_sampler(ids,
                               fold, 
                               params_normalized,
                               fold_dist,
                               i,
                               audio_path,
                               J,
                               Q,
                               ftype,
                               loss,
                               param,
                               eps) for i in idx]
    # Randomly shuffle the eds
    random.shuffle(streams)
    mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
    return pescador.maps.buffer_stream(mux, batch_size)

#for loading premade scattering features (one file the entire dataset)
@pescador.streamable    
def feature_sampler_offline(feature,gt,i):
    while True:
        yield {'input':feature.copy(),'y':gt.copy()}
        
def data_generator_offline(features,gt,batch_size,idx,active_streamers,rate,random_state=12345678):
    streams = [feature_sampler_offline(features[i,:,:].copy(),gt[i,:].copy(),i) for i in idx]
    # Randomly shuffle the seeds
    random.shuffle(streams)
    mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
    return pescador.maps.buffer_stream(mux, batch_size)

#for loading premade scattering features (one file per example)
@pescador.streamable
def feature_sampler_offsep(fold,params_normalized,idx,feature_path,J,Q,ftype,mean,var,eps):
    """
    output a {input, ground truth} pair for the designated audio sample
    load each feature from disk
    standardize them into 0 mean 1 variance, according to pre-recorded mean and variance
    logscale
    """
    i = idx

    y = params_normalized[i,:] #ground truth
    pkl_path = "_".join([ftype,"fold-"+fold,"J-" + str(J).zfill(2),"Q-" + str(Q).zfill(2)])
    pkl_name = "_".join([str(i),ftype,"fold-"+fold,"J-" + str(J).zfill(2),"Q-" + str(Q).zfill(2)+".npy"])
    fullpath = os.path.join(feature_path,ftype,pkl_path,pkl_name) 
    Sy,_ = np.load(fullpath,allow_pickle=True) #scale it per channel!!??
    #standardize along each path??
    #scale = handle_zeros_in_scale(var)
    #Sy = (Sy-mean)/scale
    #print(mean.shape,var.shape,np.mean(Sy))
    
    #logscale the input
    if eps:
        Sy = np.log1p(Sy/eps)
    while True:
        yield {'input': Sy,'y': y} #temporarily here to see if learn one task is better

def data_generator_offsep(fold,params_normalized,feature_type, mean,var,feature_path, batch_size, idx, active_streamers,
                        rate, eps,random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    streams = [feature_sampler_offsep(fold,params_normalized,i,feature_path,J,Q,ftype,mean,var,eps) for i in idx]
    # Randomly shuffle the eds
    random.shuffle(streams)
    mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
    return pescador.maps.buffer_stream(mux, batch_size)

def handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale



