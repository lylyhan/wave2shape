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

import hcqt
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
import pickle
import copy

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
import gc



import sys
sys.path.append("../src")
import cnn
import data_generator


#setup path
audio_path = "/home/han/data/drum_data/"
pkl_dir = os.path.join(audio_path,'han2022features-pkl')
random_state=44000


## helper functions


#loading and preprocessing features 
#scattering premade features
def load_scfeatures(ftype,J,Q,fold_str):
    pickle_name = "_".join([ftype,
         "fold-"+str(fold_str),"J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2)])

    pkl_path = os.path.join(pkl_dir,ftype,pickle_name + ".pkl")
    #pkl_file = open(pkl_path, 'rb')
    #Sy,y = pickle.load(pkl_file) 
    Sy,y = np.load(pkl_path,mmap_mode='r',allow_pickle=True) #free memory option 1: memmap mode
    return Sy,y

def load_scstats(ftype,J,Q,fold_str):
    pickle_name = "_".join([ftype,
         "fold-"+str(fold_str),"J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2)])

    pkl_path = os.path.join(pkl_dir,ftype,pickle_name,pickle_name + "_stats.npy")
    mean,var = np.load(pkl_path,mmap_mode='r',allow_pickle=True) #free memory option 1: memmap mode
    return mean,var

def preprocess_feat(Sy_train,Sy_test,Sy_val,eps,normalize):
    if eps:
        #logscale
        Sy_train_log = np.log1p(((Sy_train>0)*Sy_train)/eps)
        Sy_val_log = np.log1p(((Sy_val>0)*Sy_val)/eps)
        Sy_test_log = np.log1p((Sy_test>0)*Sy_test/eps)
    else:
        Sy_train_log = Sy_train
        Sy_val_log = Sy_val
        Sy_test_log = Sy_test

    if normalize:
        Sy_train_log = Sy_train_log.reshape(Sy_train_log.shape[0],Sy_train_log.shape[1]*Sy_train_log.shape[2])
        Sy_val_log = Sy_val_log.reshape(Sy_val_log.shape[0],Sy_val_log.shape[1]*Sy_val_log.shape[2])
        Sy_test_log = Sy_test_log.reshape(Sy_test_log.shape[0],Sy_test_log.shape[1]*Sy_test_log.shape[2])
        #normalize
        scaler = StandardScaler().fit(Sy_train_log)
        Sy_train_scaled = scaler.transform(Sy_train_log).reshape(Sy_train.shape)
        Sy_test_scaled = scaler.transform(Sy_test_log).reshape(Sy_test.shape)
        Sy_val_scaled = scaler.transform(Sy_val_log).reshape(Sy_val.shape)
        return Sy_train_scaled,Sy_test_scaled,Sy_val_scaled
    else:
        return Sy_train_log,Sy_test_log,Sy_val_log

#others online made features                      
def load_gt(fold,param):
    df = pd.read_csv(os.path.join(audio_path,"annotations", fold+"_param_v2.csv"))
    y = df.values[:, 1:-1]
    ids = df.values[:, 0]
    if param == "beta_alpha":
        ba = 1/y[:,-1] + 1/y[:,-1]**2
        y_ba = np.hstack((y, ba[:,None]))
        return y_ba[:,[0,1,2,3,5]], ids
    else:
        return y, ids

def preprocess_gt(y_train, y_val, y_test, param):
    if param == "beta_alpha":
        param_idx = [0,2,3,4]
    else:
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

    return y_train_normalized, y_val_normalized, y_test_normalized, scaler


## computation allocation issues
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def run_train(feature_type,
                trial,
                logscale=1e-5,
                is_normalize=True,
                is_multitask=True,
                activation="linear",
                loss="ploss",
                batch_size=64,
                n_epoch=30,
                lr=0.001,
                steps_per_epoch=50,
                predict_mode=False):
    """
    Inputs:
    feature_type: {"type":str,
                    J:int, 
                    Q: int}
    logscale: False or a float indicating the scaling factor
    batch size: batch size of training
    n_epoch: number of epoches
    steps_per_epoch: number of steps per epoch

    Outputs:
    validation loss history
    training loss history
    """
    
    #load features
    ftype = feature_type["type"]
    J = feature_type["J"]
    Q = feature_type["Q"]
    if "param" in feature_type.keys():
        param = feature_type["param"]
    else:
        param = "alpha"
    
    print("loading ground truth")
    #load, log-scale and normalize ground truth
    y_train,train_ids = load_gt("train", param)
    y_test,test_ids = load_gt("test", param)
    y_val,val_ids = load_gt("val", param)
  
    print("preprocessing ground truth")
    #normalize ground truth
    y_train_normalized,y_val_normalized,y_test_normalized,_ = preprocess_gt(y_train,y_val,y_test,param) 

   
    #designate name
    exp_type="_".join(["multitask"+str(is_multitask), activation, "log"+str(logscale)])
    
    #exp_type="_".join(["multitask"+str(is_multitask), loss, param, activation, "log"+str(logscale)])
    exp_name = "_".join([ftype,"J"+str(J),"Q"+str(Q),"bs"+str(batch_size),exp_type])
    
    #customize exp name but keep the linear activation
    if "linear" in activation:
        activation = "linear"
    
    print("running ",exp_name)
    
    if "scattering" in ftype:
        if "Q16" in exp_name and "scattering_o2" in exp_name:
            #load mean and variance
            mean,var = load_scstats(ftype,J,Q,"train")
            #print(mean,var)
            train_idx = np.arange(0,y_train.shape[0],1)
            train_batches = data_generator.data_generator_offsep(
                                                        "train",
                                                        y_train_normalized, 
                                                        feature_type,
                                                        mean,
                                                        var,
                                                        pkl_dir,
                                                        batch_size=batch_size, 
                                                        idx=train_idx,
                                                        active_streamers=32,
                                                        rate=64,
                                                        random_state=random_state,
                                                         eps=logscale)

            test_idx = np.arange(0,y_test.shape[0],1) #how long should this be?? #streamers to open
            test_batches = data_generator.data_generator_offsep(
                                                           "test",
                                                        y_test_normalized, 
                                                        feature_type,
                                                        mean,
                                                        var,
                                                        pkl_dir,
                                                        batch_size=batch_size, 
                                                        idx=test_idx,
                                                        active_streamers=32,
                                                        rate=64,
                                                        random_state=random_state,
                                                        eps=logscale)
            val_idx = np.arange(0,y_val.shape[0],1) #how long should this be?? #streamers to open
            val_batches = data_generator.data_generator_offsep(
                                                         "val",
                                                        y_val_normalized, 
                                                        feature_type,
                                                        mean,
                                                        var,
                                                        pkl_dir,
                                                        batch_size=batch_size, 
                                                        idx=val_idx,
                                                        active_streamers=32,
                                                        rate=64,
                                                        random_state=random_state,
                                                       eps=logscale)
        else:
            print("loading features")
            #load feature
            Sy_train,y_train = load_scfeatures(ftype,J,Q,"train")
            Sy_test,y_test = load_scfeatures(ftype,J,Q,"test")
            Sy_val,y_val = load_scfeatures(ftype,J,Q,"val")

            print("preprocessing features")
            #logscale and normalize features
            Sy_train_scaled,Sy_test_scaled,Sy_val_scaled = preprocess_feat(Sy_train,Sy_test,Sy_val,eps=logscale,normalize=is_normalize)
            #adjust to (naudio,nfreq,ntime)
            Sy_train_scaled = Sy_train_scaled.transpose(0,2,1)
            Sy_test_scaled = Sy_test_scaled.transpose(0,2,1)
            Sy_val_scaled = Sy_val_scaled.transpose(0,2,1)
            print("feature shapes",Sy_train_scaled.shape,Sy_test_scaled.shape,Sy_val_scaled.shape)
            #make feature generators
            print("making generators")
            train_idx = np.arange(0,y_train.shape[0],1)  # free memory option2: open less streamers
            train_batches = data_generator.data_generator_offline(Sy_train_scaled,
                                                   y_train_normalized,
                                                   batch_size=batch_size,
                                                   idx=train_idx,
                                                   active_streamers=32,
                                                   rate=64,
                                                   random_state=12345678)
            test_idx = np.arange(0,y_test.shape[0],1) 
            test_batches = data_generator.data_generator_offline(Sy_test_scaled,
                                                   y_test_normalized,
                                                   batch_size=batch_size,
                                                   idx=test_idx,
                                                   active_streamers=32,
                                                   rate=64,
                                                   random_state=12345678)
            val_idx = np.arange(0,y_val.shape[0],1) 
            val_batches = data_generator.data_generator_offline(Sy_val_scaled,
                                                   y_val_normalized,
                                                   batch_size=batch_size,
                                                   idx=val_idx,
                                                   active_streamers=32,
                                                   rate=64,
                                                   random_state=12345678)
    else: 
        #make feature generators
        train_idx = np.arange(0,y_train.shape[0],1)
        train_batches = data_generator.data_generator(train_ids,
                                                    "train",
                                                    y_train_normalized, 
                                                    feature_type,
                                                    audio_path,
                                                    batch_size=batch_size, 
                                                    idx=train_idx,
                                                    active_streamers=32,
                                                    rate=64,
                                                    random_state=random_state,
                                                     loss=loss,
                                                     param=param,
                                                     eps=logscale)

        test_idx = np.arange(0,y_test.shape[0],1) #how long should this be?? #streamers to open
        test_batches = data_generator.data_generator(test_ids,
                                                       "test",
                                                    y_test_normalized, 
                                                    feature_type,
                                                    audio_path,
                                                    batch_size=batch_size, 
                                                    idx=test_idx,
                                                    active_streamers=32,
                                                    rate=64,
                                                    random_state=random_state,
                                                    loss=loss,
                                                     param=param,
                                                    eps=logscale)
        val_idx = np.arange(0,y_val.shape[0],1) #how long should this be?? #streamers to open
        val_batches = data_generator.data_generator(val_ids,
                                                     "val",
                                                    y_val_normalized, 
                                                    feature_type,
                                                    audio_path,
                                                    batch_size=batch_size, 
                                                    idx=val_idx,
                                                    active_streamers=32,
                                                    rate=64,
                                                    random_state=random_state,
                                                    loss=loss,
                                                    param=param,
                                                   eps=logscale)


    print("making models")
    #make model
    if "scattering" in ftype:
        if "Q16" in exp_name and "scattering_o2" in exp_name:
            print(pkl_dir)
            example_str = data_generator.feature_sampler_offsep("train",y_train_normalized,0,pkl_dir,J,Q,ftype,mean,var,logscale)
            for samp in example_str.iterate():
                ex_input = samp["input"]
                break
            model = cnn.create_model_conv1d(J,Q,ex_input[None,:],activation=activation,is_multitask=is_multitask,lr=lr)
        else:
            model = cnn.create_model_conv1d(J,Q,Sy_train_scaled[0,:,:][None,:,:],activation,is_multitask=is_multitask,lr=lr)
    else:
        #create an instance of this input feature S 
        example_str = data_generator.feature_sampler(train_ids,"train",y_train_normalized,None,0,audio_path,J,Q,ftype,loss,param,logscale)
        for samp in example_str.iterate():
            ex_input = samp["input"]
            break
        model = cnn.create_model_conv2d(bins_per_oct=Q,S=ex_input[None,:],activation=activation,is_multitask=is_multitask,lr=lr)


    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpoint_path = os.path.join("../output","doce",exp_name,"trial"+str(trial))
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
    print("fitting model")
    if loss == "weighted_p":
        train_tuple = pescador.tuples(train_batches,'input','y','weights')
        val_tuple = pescador.tuples(val_batches,'input','y','weights')
        test_tuple = pescador.tuples(test_batches,'input','y','weights')
    else:
        train_tuple = pescador.tuples(train_batches,'input','y')
        val_tuple = pescador.tuples(val_batches,'input','y')
        test_tuple = pescador.tuples(test_batches,'input','y')
        
    hist = model.fit(
        train_tuple, ## add indices to inform weights
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        callbacks=[model_checkpoint_callback,reduce_lr,tensorboard_callback,recordloss_callback],
        verbose=1,
        validation_data=val_tuple,
        validation_steps=100)

    validation_loss = hist.history['val_loss']
    training_loss = hist.history['loss']
    
    print("Finished: {}".format(datetime.datetime.now()))
    print("evaluating model")
    scores = model.evaluate(test_tuple, steps=100)
    
    #optionally output prediction results
    if predict_mode:
        if ftype == "cqt":
            feat_cqt = np.load(os.path.join("/home/han/data/drum_data/han2022features-pkl/","CQT_J8_Q16_testset_features.npy"))
            if logscale:
                feat_cqt = np.log1p(feat_cqt/logscale)
            y_preds = model.predict(feat_cqt,verbose=1) 
            y_gt = y_test_normalized
        elif "scattering" in ftype:
            if Q == 16:
                feat_path = os.path.join("/home/han/data/drum_data/han2022features-pkl/",ftype,
                         ftype+"_fold-test"+"_J-"+str(J)+"_Q-"+str(Q))
                scat_feat = []
                y_gt = []
                num = 0
                for i,idx in enumerate(test_ids):
                    #idx = int(f.split("_")[0])
                    #gt_idx.append(idx)
                    Sy,y = np.load(os.path.join(feat_path,
                                                str(i)+"_"+ftype+"_fold-test"+"_J-"+str(J)+"_Q-"+str(Q)+".npy"),
                                   allow_pickle=True) #this y is not normalized 
                    y = y_test_normalized[i,:]
                    scat_feat.append(Sy)
                    y_gt.append(y)
                    num += 1 
                    if num > 500:
                        break
                scat_feat = np.stack(scat_feat)
                y_gt = np.stack(y_gt) #(501, 32, 1053), (501, 5)
            else:
                feat_path = feat_path + ".pkl"
                scat_feat,y = np.load(feat_path, allow_pickle=True)
                y_gt = y_test_normalized
              
            if logscale:
                scat_feat = np.log1p(scat_feat/logscale)

            y_preds = model.predict(scat_feat,verbose=1)
            
        pred_name = "test_preds.npy"
        np.save(os.path.join(checkpoint_path,pred_name), [y_preds, y_gt], allow_pickle=True)
        
    for val, name in zip(scores, model.metrics_names):
        print('Test {}: {:0.4f}'.format(name, val))
    
    #free up memory
    K.clear_session()
    
    del model
    #del Sy_train_scaled,Sy_test_scaled,Sy_val_scaled
    gc.collect()
    
    return validation_loss,training_loss











