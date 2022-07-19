import numpy as np
import tensorflow.keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, AveragePooling2D,Conv2D, MaxPooling2D,ReLU,Activation
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model #save and load models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.constraints import nonneg
import sys
sys.path.append("../src")
import loss



def create_model_conv1d(J,Q,S,activation,is_multitask,lr):
    """
    input dimension: (naudio,nfreq,ntime,nchan_in=1) or (naudio,nfreq,ntime,nfilter) while inside graph
    conv1d will convolve along time, kernel covers (nfilter,kernel)
        each frequency band will be transformed the same way into (naudio,new nfilter)

    """

    momentum = 0.5
    naudio,nfreq,ntime = S.shape
    nchan_in = 1  # number of input channels. 

    S_input_shape = (nfreq,ntime,nchan_in)
    S_input = keras.layers.Input(shape=S_input_shape)

    #x = BatchNormalization(momentum=momentum)(S_input)
    #first conv block
    x = Conv1D(filters=128,
        kernel_size=(8,), padding="same",name='conv1')(S_input) #FIRST LAYER 
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)  
    x = AveragePooling2D(pool_size=(1,4),padding="valid")(x) #ntime=8

    #second conv block
    x = Conv1D(filters=64,
          kernel_size=(4,),padding="same",name="conv1_2")(x) #SECOND LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)  

    #third conv block
    x = Conv1D(filters=64,
          kernel_size=(4,),padding="same",name="conv1_3")(x) #THIRD LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)  
    x = AveragePooling2D(pool_size=(1,4),padding="valid")(x) #ntime=2
 
    #fourth conv block
    x = Conv1D(filters=8,
          kernel_size=(1,),padding="same",name="conv1_4")(x) #FOURTH LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)  


    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)
    if is_multitask:
    	n_out = 5
    else:
    	n_out = 1

    x = Dense(n_out, kernel_constraint=nonneg(),bias_constraint=nonneg())(x)
    #x = BatchNormalization(momentum=momentum)(x) #this is absolutely dangerous
    x = Activation(activation)(x)  

    opt = keras.optimizers.Adam(learning_rate=lr)
    model = Model(inputs=[S_input], outputs=x)
    model.compile(loss=loss.mse_loss, optimizer=opt, metrics=[loss.mse_loss])
    #model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    return model


def create_model_conv2d(bins_per_oct,S,activation,is_multitask,lr):
    momentum=0.5
    if len(S.shape)==4:
        naudio,nfreq,ntime,nchan_in = S.shape
    elif len(S.shape)==3:
        naudio,nfreq,ntime = S.shape
        nchan_in = 1

    #batch_shape = (naudio,nfreq,ntime,nchan_in)  # USUALLY ntime=32 for T=2**11
    #S_input = S_input.reshape(batch_shape)
    S_input_shape = (nfreq,ntime,nchan_in) #(naudio,nfreq,ntime,nchan_in)
    S_input = keras.layers.Input(shape=S_input_shape)


    x = BatchNormalization(momentum=momentum)(S_input)
    #x = AveragePooling2D(pool_size=(1,8),padding="valid")(x) #ntime=32 preprocessing "QT" time dimension
    ## first conv block
    x = Conv2D(filters=128, 
        kernel_size=(bins_per_oct,8), padding="same",kernel_regularizer=None,name='conv1')(x) #FIRST LAYER:one oct along freq, 8 time bins
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)
    ## pool across time
    x = AveragePooling2D(pool_size=(1,8),padding="valid")(x) #ntime=8
    
    ## second conv block
    x = Conv2D(filters=64,
          kernel_size=(bins_per_oct,4),padding="same",kernel_regularizer=None,name="conv1_2")(x)#SECOND LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)
    
    ## third conv block
    x = Conv2D(filters=64,
          kernel_size=(bins_per_oct,4),padding="same",kernel_regularizer=None,name="conv1_3")(x)#THIRD LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)
    
    ## pool across time
    x = AveragePooling2D(pool_size=(1,8),padding="valid")(x) #ntime=2
    

    ## fourth conv block
    x = Conv2D(filters=8,
          kernel_size=(bins_per_oct,1),padding="same",kernel_regularizer=None,name="conv1_4")(x) #FOURTH LAYER
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)

    ## flattening and dense layers
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=None)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Activation("relu")(x)
  
    if is_multitask:
        n_out = 5
    else:
        n_out = 1

    x = Dense(n_out, kernel_regularizer=None, kernel_constraint=nonneg(),bias_constraint=nonneg())(x)
    #x = BatchNormalization(momentum=momentum)(x) #this batch norm will shift output to zero mean?
    x = Activation(activation)(x)

    opt = keras.optimizers.Adam(learning_rate=lr)
    model = Model(inputs=[S_input], outputs=x)
    model.compile(loss=loss.mse_loss, optimizer=opt, metrics=[loss.mse_loss])
    #model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    return model



def create_model_adjustable(J,Q,N,order,k_size,nchan_out,activation,Sy,lr):
    """
    input dim: (naudio, ntime, nfreq)
    conv1d will convolve along time dimension, the kernel covers (nfreq, kernel)
        thus all frequencies at each time step will be summarized into one value per filter

    """
    ntime,nfreq = Sy.shape
    input_shape = Sy.shape
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
                     kernel_size=kernel_size, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        #print('before pool',model.layers[-1].output_shape)
        if model.layers[-1].output_shape[1] >= 4:
            model.add(AveragePooling1D(pool_size=(4,)))
        elif model.layers[-1].output_shape[1] == 2:
            model.add(AveragePooling1D(pool_size=(2,)))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    #what activation should be chosen for last layer, for regression problem? should be a linear function
    model.add(Dense(5, activation=activation)) #output layer that corresponds to the 5 physical parameters.
    

    opt = keras.optimizers.Adam(learning_rate=lr)
    # Compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])




    return model





