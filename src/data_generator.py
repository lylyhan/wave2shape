import librosa
import pescador
import features
import soundfile as sf
import numpy as np
import random
import os

audio_path = "/home/han/data/drum_data/"


@pescador.streamable
def feature_sampler(ids, fold, params_normalized, fold_dist, weight_type, idx, audio_path, J, Q, ftype, loss, param, eps, pitchmode):
    """
    output a {input, ground truth} pair for the designated audio sample
    """
    i = idx
    y = params_normalized[i,:] #ground truth
   
    #load weights here!
    if fold_dist is None:
        weights = np.ones((5,)) #does not affect the following computation
    else:
        if weight_type == "nn_limit":
            weights = fold_dist[i,:] #logscaling the weights will largely diminish the penalization
        elif weight_type == "grad": #JTJ
            weights = np.sqrt(np.diagonal(fold_dist[i,:,:]))
        elif weight_type == "riemann": #volumetric sample weight
            M = fold_dist[i,:,:]
            w,v = np.eig(M)
            weights = np.sqrt(np.prod(w))
            
    
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
    if pitchmode == "wpitch":
        idrange = 0
    else:
        idrange = 1
    if loss == "weighted_p":
        if weight_type == "riemann":
            while True:
                yield {'input': Sy,'y': y[idrange::],"weights":weights[idrange::, None],
                      "quadratic":M} #is the fourth key allowed??
        else:
            while True:
                yield {'input': Sy,'y': y[idrange::],"weights":weights[idrange::, None]} #third key is supposed to be the sample weight!!
    else:
        while True:
            yield {'input': Sy,'y': y[idrange::]}

def data_generator(ids,fold,params_normalized,feature_type,audio_path, batch_size, idx, active_streamers,
                        rate,loss, weight_setup, param, eps, pitchmode,random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    #load the weights npy file
    if loss == "ploss":
        fold_dist = None
        weight_type = None
    else:
        if weight_setup is None:
            raise NameError('weighted_ploss has to have weight_setup')
        else:
            weight_type = weight_setup["type"]
            weight_nnbr = weight_setup["n_nbr"]
            weight_n = weight_setup["n"]
            if weight_type == "nn_limit":
                fold_dist = np.load(os.path.join(audio_path, fold + '_nbr_dist_nnbr'+str(weight_nnbr) + '_n' + str(weight_n) + 'jtfsavgf_nnadvanced.npy'))
                if param == "alpha":
                    fold_dist = fold_dist[:,:-1]
                else:
                    fold_dist = fold_dist[:,[0,1,2,3,5]]
            elif weight_type == "grad":
                fold_dist = np.load(os.path.join(audio_path, fold + '_grad_jtfs.npy'))
            elif weight_type == "riemann":
                fold_dist = np.load(os.path.join(audio_path, fold + '_grad_jtfs.npy'))
            
    streams = [feature_sampler(ids,
                               fold, 
                               params_normalized,
                               fold_dist,
                               weight_type,
                               i,
                               audio_path,
                               J,
                               Q,
                               ftype,
                               loss,
                               param,
                               eps,
                               pitchmode) for i in idx]
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
def feature_sampler_offsep(ids,fold,params_normalized,idx,feature_path,J,Q,ftype,eps):
    """
    output a {input, ground truth} pair for the designated audio sample
    load each feature from disk
    standardize them into 0 mean 1 variance, according to pre-recorded mean and variance
    logscale
    """
    i = idx
    y = params_normalized[i,:] #ground truth
    pkl_path = "_".join([ftype,"fold-"+fold,"J-" + str(J).zfill(2),"Q-" + str(Q).zfill(2)])
    pkl_name = "_".join([str(ids[i]),ftype,"fold-"+fold,"J-" + str(J).zfill(2),"Q-" + str(Q).zfill(2)+".npy"])
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

def data_generator_offsep(ids,fold,params_normalized,feature_type, feature_path, batch_size, idx, active_streamers,
                        rate, eps,random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    streams = [feature_sampler_offsep(ids,fold,params_normalized,i,feature_path,J,Q,ftype,eps) for i in idx]
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



