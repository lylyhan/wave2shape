import numpy as np
import tensorflow.keras
import torch
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Conv2D, MaxPooling2D,ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model #save and load models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from kymatio import Scattering1D
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pescador
import random
import os
import librosa
import sys
import datetime
import  time
import argparse
import pickle

def getsc_new(y,J,Q_num,order):
    """
    this function outputs scattering transform of a time-domain signal.
    """
    N = len(y)
    scattering = Scattering1D(J = J,shape=(N,), Q = Q_num, max_order=order)
    Sy = scattering(torch.Tensor(y))
    return Sy


def feature_sampler(df,params_normalized,idx,path_to_folder,J,Q,order):
    """
    output a {input, ground truth} pair for the designated audio sample
    """
    i=idx
    y=np.array(params_normalized[i,:]).reshape((5,)) #df.values[i,1:-1]
    path_to_audio = os.path.join(path_to_folder,str(df.values[i,0])+"_sound.wav") 
    x,fs=librosa.load(path_to_audio)
    Sy = getsc_new(x,J,Q,order)
    m,n = Sy.shape
    Sy2 = np.array(Sy).reshape((n,m))
    
    while True:
        yield {'input': Sy2,'y': y}

        
def data_generator(df, params_normalized, path_to_folder, J, Q, order, batch_size, idx, active_streamers,
                        rate, random_state=12345678):
    """
    use streamers to output a batch of {input groundtruth} pairs. 
    """
    seeds = []
    for i in idx:
        streamer = pescador.Streamer(feature_sampler, df, params_normalized, i,path_to_folder,J,Q,order)
        seeds.append(streamer)

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    mux = pescador.StochasticMux(seeds, active_streamers, rate=rate, random_state=random_state)
   
    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)

def train(epochs,batch_size,active_streamers,J,Q,order,patience):

    df_train = pd.read_csv("../notebooks/train_param.csv")
    df_test = pd.read_csv("../notebooks/test_param.csv")
    df_val = pd.read_csv("../notebooks/val_param.csv")
    df_full = pd.read_csv("../notebooks/diffshapes_param.csv")


    # normalization of the physical parameters
    params = df_train.values[:,1:-1]
    scaler = MinMaxScaler()
    scaler.fit(params)
    train_params_normalized = scaler.transform(params)
    test_params_normalized = scaler.transform(df_test.values[:,1:-1])
    val_params_normalized = scaler.transform(df_val.values[:,1:-1])

    ## first run with small number of training
    #epochs=12
    #batch_size=32
    random_state=12345678
    #active_streamers=64
    path_to_train = "/scratch/hh2263/drum_data/train/"
   #path_to_test = "/scratch/hh2263/drum_data/test/"
    #J = 8
    #Q = 1
    #order = 2 # remember to go to order 2 eventually
    train_idx = np.arange(0,params.shape[0],1)#np.arange(0,1000,1) #df_train.values[:1000,0]
    train_batches=data_generator(df_train,train_params_normalized, path_to_train,J, Q, order, batch_size, train_idx,active_streamers,rate=64,random_state=random_state)
    steps_per_epoch = len(train_idx) // batch_size


    ##build the model
    fname = random.choice(os.listdir(path_to_train))
    rand_audio = os.path.join(path_to_train,fname)
    y,sr = librosa.load(rand_audio)
    Sy = getsc_new(torch.Tensor(y),J,Q,order).T
    nrow, ncol = Sy.shape 
    naudio = batch_size         # number of images in batch
    nchan_in = 1       # number of input channels.  1 since it is BW
    input_shape = Sy.shape


    kernel_size = (8,)
    nchan_out = 16

    K.clear_session()
    model=Sequential()
    model.add(Conv1D(input_shape=input_shape, filters=nchan_out,
                     kernel_size=kernel_size,activation= "relu", padding="same",name='conv1'))
    model.add(AveragePooling1D(pool_size=(4,)))
    model.add(Conv1D(filters=16,
                     kernel_size=kernel_size,activation= "relu", padding="same",name='conv2' ))
    model.add(AveragePooling1D(pool_size=(4,)))
    model.add(Conv1D(filters=16,
                     kernel_size=kernel_size,activation= "relu", padding="same",name='conv3' ))
    model.add(AveragePooling1D(pool_size=(4,)))
    model.add(Conv1D(filters=16,
                     kernel_size=kernel_size,activation= "relu", padding="same",name='conv4' ))
    model.add(AveragePooling1D(pool_size=(2,)))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    #what activation should be chosen for last layer, for regression problem? should be a linear function
    model.add(Dense(5, activation='linear')) #output layer that corresponds to the 5 physical parameters.


    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    output_dir = "../output/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filepath = os.path.join(output_dir, 'model.h5')
    log_filepath = os.path.join(output_dir, 'train_log.csv')

    callbacks = []
    callbacks.append(EarlyStopping(patience=10))
    callbacks.append(ModelCheckpoint(model_filepath, save_best_only=True))
    callbacks.append(CSVLogger(log_filepath))

    print("Fitting model.")
    sys.stdout.flush()

    #load validation features+gt 
    pkl_path = '/scratch/hh2263/drum_data/val/J_8_Q_1_order_2.pkl'
    pkl_file = open(pkl_path, 'rb')
    Sy_val,y_val = pickle.load(pkl_file) 
    Sy_val = Sy_val.reshape((Sy_val.shape[2],Sy_val.shape[0],Sy_val.shape[1]))
    y_val = y_val.astype('float32')

    print("Validation set dimension is "+str(Sy_val.shape)+" and "+str(y_val.shape))

    train_gen = pescador.maps.keras_tuples(train_batches, 'input', 'y')
    #preliminary test
    for epoch in range(epochs):
        model.fit(train_gen,steps_per_epoch=steps_per_epoch,epochs=1)
        print('done fitting')
        loss,accuracy = model.evaluate(Sy_val,y_val)
        print(loss,accuracy)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('epochs',type=int,default=12)
    parser.add_argument('batch_size',type=int,default=32)
    parser.add_argument('active_streamers',type=int,default=64)
    parser.add_argument('J',type=int,default=8)
    parser.add_argument('Q',type=int,default=1)
    parser.add_argument('order',type=int,default=2)
    parser.add_argument('patience',type=int,default=10)

    args = vars(parser.parse_args())

    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print("Generating validation pickle files.")
    #START TRAINING
    train(**args)

    print(str(datetime.datetime.now()) + " Finish.")

    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(elapsed_hours,
    elapsed_minutes,
    elapsed_seconds)
    print("Total elapsed time: " + elapsed_str + ".")


