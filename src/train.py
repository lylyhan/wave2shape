import numpy as np
import tensorflow.keras
from tensorflow import keras
import torch
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Conv2D, MaxPooling2D,ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model #save and load models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import IPython.display as ipd
from kymatio import Scattering1D
import hitdifferentparts
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pescador
import random
import os
import librosa
import pickle
import matplotlib.pyplot as plt
import math



# Parse input arguments.
args = sys.argv[1:]
J = int(args[0])
order = int(args[1])
Q = int(args[2])


#make the model
#zoom factor can only be 1/4, 1/2
def create_model_adjustable(J,Q,order,k_size,nchan_out,activation):
    N = 2**15
    y = np.random.rand(N)
    scattering = Scattering1D(J = J,shape=(N,), Q = Q, max_order=order)
    Sy = np.array(scattering(torch.Tensor(y))).T
    input_x,input_y = Sy.shape
    nchan_in = 1       # number of input channels.  1 since it is BW
  
    input_shape = (input_x,input_y)#Sy.shape
	kernel_size = (k_size,)
	K.clear_session()
	model=Sequential()
	#1 conv layer +  1 batch normalization + nonlinear activation + pooling
	model.add(BatchNormalization(input_shape=input_shape))
	model.add(Conv1D(filters=nchan_out,
	                 kernel_size=kernel_size, padding="same",name='conv1'))
	#model.add(BatchNormalization())
	model.add(Activation("relu"))

	if model.layers[-1].output_shape[1]>=4:
	    pool = 4
	elif model.layers[-1].output_shape[1]==2:
	    pool = 2
	    
	model.add(AveragePooling1D(pool_size=(pool,)))


	for i in range(3):
	    model.add(Conv1D(filters=nchan_out,
	                 kernel_size=kernel_size, padding="same" ))
	    model.add(BatchNormalization())
	    model.add(Activation("relu"))
	    #print('before pool',model.layers[-1].output_shape)
	    if model.layers[-1].output_shape[1] >= 4:
	        model.add(AveragePooling1D(pool_size=(4,)))
	    elif model.layers[-1].output_shape[1] == 2:
	        model.add(AveragePooling1D(pool_size=(2,)))
	    #print(model.layers[-1].output_shape)

	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	#what activation should be chosen for last layer, for regression problem? should be a linear function
	model.add(Dense(5, activation='linear')) #output layer that corresponds to the 5 physical parameters.


	# Compile the model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])




    return model

pkl_dir = '/scratch/hh2263/drum_data/han2020fa_sc-pkl/'
#J = 8
#Q = 1
#order = 2
pickle_name = "_".join(
    ["scattering",
     "J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2), "order" + str(order),"fold-"+str(fold_str)]
)

pkl_path_train = os.path.join(pkl_dir,pickle_name+"fold-train.pkl")
pkl_train = open(pkl_path_train, 'rb')
Sy_train,y_train = pickle.load(pkl_train) 

pkl_path_val = os.path.join(pkl_dir,pickle_name+"fold-val.pkl")
pkl_val = open(pkl_path_val,'rb')
Sy_val,y_val = pickle.load(pkl_val)

pkl_path_test = os.path.join(pkl_dir,pickle_name+"fold-test.pkl")
pkl_test = open(pkl_path_test,'rb')
Sy_test,y_test = pickle.load(pkl_test)

#log scale p and D
for idx in range(2,4):
    y_train[:,idx] = [math.log10(i) for i in y_train[:,idx]]
    y_test[:,idx] = [math.log10(i) for i in y_test[:,idx]]
    y_val[:,idx] = [math.log10(i) for i in y_val[:,idx]]

#df_train = pd.read_csv("../notebooks/train_param.csv")
#df_test = pd.read_csv("../notebooks/test_param.csv")
#df_val = pd.read_csv("../notebooks/val_param.csv")
#df_full = pd.read_csv("../notebooks/diffshapes_param.csv")

#params = df_train.values[:,1:-1]
#for idx in range(2,4):
#    params[:,idx] = [math.log10(i) for i in params[:,idx]]

# normalization of the physical parameters
scaler = MinMaxScaler()
scaler.fit(y_train)
y_train_normalized = scaler.transform(y_train)
y_val_normalized = scaler.transform(y_val)
y_test_normalized = scaler.transform(y_test)

#log scale the input
eps = 1e-11
Sy_train_log2 = np.log1p(((Sy_train>0)*Sy_train)/eps)
Sy_val_log2 = np.log1p(((Sy_val>0)*Sy_val)/eps)
Sy_test_log2 = np.log1p((Sy_test>0)*Sy_test/eps)


#train the model


trial_dir = "../output/tests/"
os.makedirs(trial_dir, exist_ok=True)
best_validation_loss = np.inf
zoom_factor = 1
n = Sy_train.shape[0]
shape_time = round(Sy_train.shape[1] * zoom_factor)
steps_per_epoch = 50
bs = 64
m = bs*steps_per_epoch
idx = np.arange(0,n,1)
val_loss=[]
train_loss = []
model_adjustable = create_model_adjustable(J=J,Q=Q,order=order,k_size=8,nchan_out=16,activation='linear')
#model_adjustable.summary()
for epoch in range(30):
    np.random.shuffle(idx)
    Sy_temp = Sy_train_log2[idx[:m],:shape_time,:]
    y_temp = y_train_normalized[idx[:m],:]
    
    hist = model_adjustable.fit(Sy_temp,
                y_temp,
                epochs=1,
                verbose=2,
                batch_size=bs,
                validation_data = (Sy_val_log2[:,-shape_time:,:],y_val_normalized),
                use_multiprocessing=False)
    validation_loss = hist.history['val_loss'][0]
    val_loss.append(validation_loss)
    train_loss.append(hist.history['loss'][0])
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        #epoch_str = "epoch-" + str(epoch).zfill(3)
        epoch_network_path = os.path.join(
           trial_dir, "_".join([ "J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2), "order" + str(order)]) + ".h5")
        model.save(epoch_network_path)
        

















