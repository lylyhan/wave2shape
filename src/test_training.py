import pescador
import numpy as np
import keras
import pandas as pd
import torch
import os
import math
from kymatio.torch import Scattering1D
import sys
sys.path.append("../src")
import cnn
import ftm_ver2 as ftm2
import hcqt
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
import pickle

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard,CSVLogger
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Conv2D, MaxPooling2D,ReLU,AveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model #save and load models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
import librosa
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datetime




audio_path = "/home/han/data/drum_data/"
pickle_dir = os.path.join(audio_path, "han2022features-pkl")
random_state=44000
batch_size=64
n_epoch=30
steps_per_epoch=50 

@pescador.streamable
def feature_sampler(ids,fold,params_normalized,idx,audio_path,J,Q,ftype):
    """
    output a {input, ground truth} pair for the designated audio sample
    """
    i = idx
    y = params_normalized[i,:] #ground truth
    fullpath = os.path.join(audio_path,fold,str(ids[i])+"_sound.wav") 
    x,sr = sf.read(fullpath)
    fmin = 0.4*sr*2**(-J)
    if ftype == "cqt":
        Sy = make_cqt(x,b=Q,sr=sr,n_oct=J,fmin=fmin)
        nfreq,ntime = Sy.shape
        Sy = Sy[:,:,None]
    elif ftype == "vqt":
        Sy = make_vqt(x,b=Q,sr=sr,n_oct=J,fmin=fmin)
        nfreq,ntime = Sy.shape
        Sy = Sy[:,:,None]
    elif ftype == "hcqt":
        Sy = make_hcqt(x,b=Q,sr=sr,n_oct=J-2,fmin=fmin)
        nharm,nfreq,ntime = Sy.shape
        Sy = Sy.reshape((nfreq,ntime,nharm))
    #logscale the input
    eps = 1e-11
    Sy = np.log1p(Sy/eps)
    while True:
        yield {'input': Sy,'y': y} #temporarily here to see if learn one task is better

def data_generator(ids,fold,params_normalized,feature_type,audio_path, batch_size, idx, active_streamers,
                        rate, random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    streams = [feature_sampler(ids,fold,params_normalized,i,audio_path,J,Q,ftype) for i in idx]
    # Randomly shuffle the seeds
    random.shuffle(streams)
    mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
    return pescador.maps.buffer_stream(mux, batch_size)

@pescador.streamable    
def feature_sampler_offline(feature,gt,i):
    while True:
        yield {'input':feature,'y':gt}
        
def data_generator_offline(features,gt,batch_size,idx,active_streamers,rate,random_state=12345678):
    streams = [feature_sampler_offline(features[i,:,:],gt[i,:],i) for i in idx]
    # Randomly shuffle the seeds
    random.shuffle(streams)
    mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
    return pescador.maps.buffer_stream(mux, batch_size)


def load_gt(fold):
	df = pd.read_csv(os.path.join(audio_path,"annotations", fold+"_param_v2.csv"))
	y = df.values[:, 1:-1]
	ids = df.values[:, 0]
	return y,ids

def preprocess_gt(y_train, y_val, y_test):
	#logscale
	for idx in [0,2,3]:
		y_train[:,idx] = [math.log10(i) for i in y_train[:,idx]]
		y_test[:,idx] = [math.log10(i) for i in y_test[:,idx]]
		y_val[:,idx] = [math.log10(i) for i in y_val[:,idx]]
	#normalize
	scaler = MinMaxScaler()
	scaler.fit(y_train)
	y_train_normalized = scaler.transform(y_train)
	y_val_normalized = scaler.transform(y_val)
	y_test_normalized = scaler.transform(y_test)

	return y_train_normalized,y_val_normalized,y_test_normalized

def get_feat(ftype,fold_str,J,Q,pickle_dir):
    pickle_name = "_".join([ftype,"fold-"+str(fold_str),"J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2)])
    pickle_path = os.path.join(pickle_dir,ftype,pickle_name + ".pkl")
    reader = open(pickle_path,'rb')
    Sy,y = pickle.load(reader) 
    return Sy,y

def preprocess_feat(Sy_train,Sy_test,Sy_val,eps,normalize):
    #logscale
    Sy_train_log = np.log1p(((Sy_train>0)*Sy_train)/eps).reshape(Sy_train.shape[0],
                                                                Sy_train.shape[1]*Sy_train.shape[2])
    Sy_val_log = np.log1p(((Sy_val>0)*Sy_val)/eps).reshape(Sy_val.shape[0],
                                                          Sy_val.shape[1]*Sy_val.shape[2])
    Sy_test_log = np.log1p((Sy_test>0)*Sy_test/eps).reshape(Sy_test.shape[0],
                                                           Sy_test.shape[1]*Sy_test.shape[2])
    if normalize:
        #normalize
        scaler = StandardScaler().fit(Sy_train_log)
        Sy_train_scaled = scaler.transform(Sy_train_log).reshape(Sy_train.shape)
        Sy_test_scaled = scaler.transform(Sy_test_log).reshape(Sy_test.shape)
        Sy_val_scaled = scaler.transform(Sy_val_log).reshape(Sy_val.shape)
    
        return Sy_train_scaled,Sy_test_scaled,Sy_val_scaled
    else:

        return Sy_train_log.reshape(Sy_train.shape),Sy_test_log.reshape(Sy_test.shape),Sy_val_log.reshape(Sy_val.shape)

                           
    
#Make CQT
def make_cqt(waveform,b,sr,n_oct,fmin):
	Cx = librosa.cqt(waveform,sr=sr,n_bins=(n_oct)*b,fmin=fmin,hop_length=256,bins_per_octave=b) 
	return Cx

#Make HCQT
def make_hcqt(waveform,b,sr,n_oct,fmin):
	comp_hcqt = hcqt.HCQT(sr,bins_per_octave=b,harmonics=[0.5,1,2,3,4,5],
                 n_octaves=n_oct-2,f_min=fmin,hop_length=256) #choice of hop size?
	h_cqt = comp_hcqt.compute_hcqt(waveform,sr)

	return h_cqt

#Make VQT
def make_vqt(waveform,b,sr,n_oct,fmin):
	Vx = librosa.vqt(waveform,sr=sr,n_bins=(n_oct)*b,fmin=fmin,hop_length=256,bins_per_octave=b) 
	return Vx



def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    



if __name__ == "__main__":
    fix_gpu()
    ftype="scattering_o2"
    J = 12
    Q = 16
    bs = batch_size
    centroid = False
    time_average=False
    activation="sigmoid"
    exp_type="multitask_sigact" #"omegaonly","omegaonly_centroid","multitask"
    exp_name = "_".join([ftype,"J"+str(J),"Q"+str(Q),"bs"+str(batch_size),exp_type])
    print(exp_name)
    
    #make or load features:
    if "scattering" in ftype:
        Sy_train,y_train = get_feat(ftype,"train",J,Q,pickle_dir)
        Sy_test, y_test = get_feat(ftype,"test",J,Q,pickle_dir)
        Sy_val, y_val = get_feat(ftype,"val",J,Q,pickle_dir)
        #normalize ground truth
        y_train_normalized,y_val_normalized,y_test_normalized = preprocess_gt(y_train,y_val,y_test) 
        #logscale and normalize features
        Sy_train_scaled,Sy_test_scaled,Sy_val_scaled = preprocess_feat(Sy_train,Sy_test,Sy_val,eps=1e-3,normalize=False)
        #model = cnn.create_model_conv1d(J,Q,Sy_train_scaled[0,:,:][None,:,:],activation)
        #print(model.summary())
        model = cnn.create_model_adjustable(J=J,Q=Q,order=1,k_size=8,nchan_out=16,activation='linear',S=Sy_train_scaled[0,:,:])
        print(model.summary())
        #reshape the features to (naudio,freq,time)
        if not "oldpaper" in exp_type:
            Sy_train_scaled = Sy_train_scaled.transpose(0,2,1)
            Sy_test_scaled = Sy_test_scaled.transpose(0,2,1)
            Sy_val_scaled = Sy_val_scaled.transpose(0,2,1)
        #make feature generators
        train_idx = np.arange(0,y_train.shape[0],1) 
        train_batches = data_generator_offline(Sy_train_scaled,
                                               y_train_normalized,
                                               batch_size=bs,
                                               idx=train_idx,
                                               active_streamers=32,
                                               rate=64,
                                               random_state=12345678)
        test_idx = np.arange(0,y_test.shape[0],1) 
        test_batches = data_generator_offline(Sy_test_scaled,
                                               y_test_normalized,
                                               batch_size=bs,
                                               idx=test_idx,
                                               active_streamers=32,
                                               rate=64,
                                               random_state=12345678)
        val_idx = np.arange(0,y_val.shape[0],1) 
        val_batches = data_generator_offline(Sy_val_scaled,
                                               y_val_normalized,
                                               batch_size=bs,
                                               idx=val_idx,
                                               active_streamers=32,
                                               rate=64,
                                               random_state=12345678)
        
    else:
        y_train,train_ids = load_gt("train")
        y_test,test_ids = load_gt("test")
        y_val,val_ids = load_gt("val")
        #log scale w, p and D, normalize
        y_train_normalized,y_val_normalized,y_test_normalized = preprocess_gt(y_train,y_val,y_test)
        train_idx = np.arange(0,y_train.shape[0],1) #how long should this be?? #streamers to open
        train_batches = data_generator(train_ids,
                                       "train",
                                    y_train_normalized, 
                                    {"type":"cqt","J":12,"Q":16},
                                    audio_path,
                                    batch_size=bs, 
                                    idx=train_idx,
                                    active_streamers=32,
                                    rate=64,
                                    random_state=random_state)

        test_idx = np.arange(0,y_test.shape[0],1) #how long should this be?? #streamers to open
        test_batches = data_generator(test_ids,
                                       "test",
                                    y_test_normalized, 
                                    {"type":"cqt","J":12,"Q":16},
                                    audio_path,
                                    batch_size=bs, 
                                    idx=test_idx,
                                    active_streamers=32,
                                    rate=64,
                                    random_state=random_state)
        val_idx = np.arange(0,y_val.shape[0],1) #how long should this be?? #streamers to open
        val_batches = data_generator(val_ids,
                                     "val",
                                    y_val_normalized, 
                                    {"type":"cqt","J":12,"Q":16},
                                    audio_path,
                                    batch_size=bs, 
                                    idx=val_idx,
                                    active_streamers=64,
                                    rate=64,
                                    random_state=random_state)
    
        example_str = feature_sampler(train_ids,"train",y_train_normalized,0,audio_path,J,Q,ftype)
        for samp in example_str.iterate():
            ex_input = samp["input"]
            break
        print("input shape",ex_input.shape)
        sr = 22050
        if "multitask" in exp_name:
            model = cnn.create_model_conv2d(n_oct=J,bins_per_oct=Q,S=ex_input[None,:])
        else:     
            freqs = librosa.cqt_frequencies(n_bins=J*Q,fmin=0.4*sr*2**(-J),bins_per_octave=Q)
            #sprint("centroid" in exp_name)
            model = cnn.create_model_conv2d_onetask(n_oct=J,bins_per_oct=Q,S=ex_input[None,:],is_centroid=centroid,
                                                    time_average=time_average,vector=freqs,activation=activation)

    print(model.summary())
    print("model made, start training")
    #callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpoint_path = os.path.join("./",exp_name,'best_models')
    os.makedirs(checkpoint_path,exist_ok=True)
    
    model_checkpoint_callback = ModelCheckpoint(
                                filepath=os.path.join(checkpoint_path,"ckpt"),
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='max',
                                save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0005)
    tensorboard_callback = TensorBoard(log_dir=os.path.join(checkpoint_path,"ts_board_logs"),
                                      histogram_freq=1) #see with  "=tensorboard --logdirpath_to_your_logs"
    recordloss_callback = CSVLogger(os.path.join(checkpoint_path,"training.log"), separator=",", append=False)

    model.fit(
        pescador.tuples(train_batches,'input','y'),
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        callbacks=[model_checkpoint_callback,reduce_lr,tensorboard_callback,recordloss_callback],
        verbose=1,
        validation_data=pescador.tuples(val_batches,'input','y'),
        validation_steps=100)

    print("Finished: {}".format(datetime.datetime.now()))
    scores = model.evaluate(pescador.tuples(test_batches,'input','y'),steps=100)
    for val, name in zip(scores, model.metrics_names):
        print('Test {}: {:0.4f}'.format(name, val))
