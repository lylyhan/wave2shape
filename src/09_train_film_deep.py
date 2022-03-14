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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pescador
import random
import os
import librosa
import pickle
import math
import sys


tf.enable_eager_execution()

# Parse input arguments.
args = sys.argv[1:]
J = int(args[0])
order = int(args[1])
Q = 1

index = int(args[2])
eps = 10**(-index)
trial = int(args[3])

depth = int(args[4])
units = int(args[5])

#deep layer - aggregate depth inside (weighted sum of a_{p,l} as gamma) alternatively make F as fully connected layer
#deep layer the right way - hidden layers before modulating
class FiLM(keras.layers.Layer):

    def __init__(self, depth=2, units=16, **kwargs):
        self.depth = depth
        self.units = units
        super(FiLM, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'depth': self.depth,
            'units': self.units
        })
        return config

    def build(self, input_shape): # input shape = (None,t,p,2)
       
        Sc_input_shape, u_input_shape = input_shape
        # print(Sc_input_shape)
        self.height, self.width = Sc_input_shape[1:] # t, p , 2
        FiLM_tns = u_input_shape[1]
        # print(FiLM_tns,u_input_shape)
        self.n_feature_maps = self.width

        u_shape = tf.TensorShape(u_input_shape)
        last_dim = tf.dimension_value(u_shape[-1])

        if self.depth > 0:
            self.dense={}
            #add d layers of hidden units
            for d in range(self.depth):
                if d > 0:
                    last_dim = self.units

                self.dense[d] = self.add_weight(name = 'dense_'+str(d),
                                              shape = (last_dim, self.units),
                                              initializer='normal',trainable=True)
        else:
            self.units = FiLM_tns
    
        #last layer's weight 
        self.kernel = self.add_weight(name = 'kernel', 
                                      shape = (self.units, int(2*self.n_feature_maps)),
                                      initializer = 'normal', trainable = True) 
        
        
        #assert(int(2 * self.n_feature_maps)==FiLM_tns_shape[1]) #film tensor size need to be len(u) by 2p 
        super(FiLM, self).build(input_shape)

    def call(self, x):
        #assert isinstance(x, list)
        conv_output, FiLM_tns = x # x = [Sx,u]; [t by p, length-2]
        #FiLM.append(0) # to include the bias term
        
        if self.depth > 0:
            for d in range(self.depth):
                #FiLM_tns = core_ops.dense(FiLM_tns,self.dense[d])
                FiLM_tns = K.dot(FiLM_tns,self.dense[d])
                #print(FiLM_tns.shape,self.dense[d].shape)

        FiLM_tns_agg = K.dot(FiLM_tns,self.kernel) # making each a_{p,l},b_{p,l}
            #FiLM_tns_agg = FiLM_tns_agg + FiLM_tns
        #aggregate together with weights
        
        #put [f(u),g(u)] in the fourth dimension
        FiLM_tns = K.expand_dims(FiLM_tns_agg, axis=[1]) 
        #FiLM_tns = K.expand_dims(FiLM_tns, axis=[1]) #make it into [1, 1, 1,2p]
        FiLM_tns = K.tile(FiLM_tns, [1, self.height, 1]) #[1,Sx.shape[0],Sx.shape[1],2p]

        #extract f(u) and g(u)
        gammas = FiLM_tns[ :, :, :self.n_feature_maps] 
        betas = FiLM_tns [ :, :, self.n_feature_maps:]
        
        # Apply affine transformation
        return (1 + gammas) * conv_output + betas


    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        #return (input_shape[1],input_shape[2],self.n_feature_maps) # 
        return input_shape[:-1]


    


#make the model
#zoom factor can only be 1/4, 1/2
def create_model_adjustable(J,Q,order,k_size,nchan_out,activation,depth, units):
    N = 2**15
    y = np.random.rand(N)
    scattering = Scattering1D(J = J,shape=(N,), Q = Q, max_order=order) 
    Sy = np.array(scattering(torch.Tensor(y))).T
    nchan_in = 1       # number of input channels.  1 since it is BW

    #initialize input sizes
    Sc_input_shape = Sy.shape
    u_input_shape = (2,)

    
    input_shape = [Sc_input_shape,u_input_shape]#Sy.shape
    kernel_size = (k_size,)
    
    
    K.clear_session()
    #define input
    Sc_input = keras.layers.Input(shape=Sc_input_shape)
    u_input = keras.layers.Input(shape=u_input_shape)
    
    Sc_input_batched = BatchNormalization()(Sc_input)

    x = FiLM(input_shape = [Sc_input_shape,u_input_shape],name='FiLM_layer',
            dynamic=True, depth=depth, units=units)([Sc_input_batched,u_input])
    
    #1 conv layer +  1 batch normalization + nonlinear activation + pooling
    #x = BatchNormalization()(x[0])
    x = Conv1D(filters=nchan_out,
        kernel_size=kernel_size, padding="same",name='conv1')(x[0])
    x = Activation("relu")(x)


    if x[0].shape[1]>=4:
        pool = 4
    elif x[0].shape[1]==2:
        pool = 2

    x = AveragePooling1D(pool_size=(pool,))(x)

    for i in range(3):

        x = Conv1D(filters=nchan_out,
                     kernel_size=kernel_size, padding="same" )(x)  
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
       
        if x.shape[1] >= 4:
            x = AveragePooling1D(pool_size=(4,))(x)
          
        elif x.shape[1] == 2:
            x = AveragePooling1D(pool_size=(2,))(x)

    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    #what activation should be chosen for last layer, for regression problem? should be a linear function
    x = Dense(5, activation=activation)(x)


    # Compile the model
    model = Model(inputs=[Sc_input, u_input], outputs=x)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model

pkl_dir = '/scratch/hh2263/drum_data_ver2/drumv2_sc-pkl/'
#J = 8
#Q = 1
#order = 2
pickle_name = "_".join(
    ["scattering",
    "J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2), "order" + str(order)]
)

pkl_half = "_".join(
    ["_J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2), "order" + str(order)]
)

pkl_path_train = os.path.join(pkl_dir,"scattering_fold-train"+pkl_half+".pkl")
pkl_train = open(pkl_path_train, 'rb')
Sy_train,y_train = pickle.load(pkl_train) 

pkl_path_val = os.path.join(pkl_dir,"scattering_fold-val"+pkl_half+".pkl")
pkl_val = open(pkl_path_val,'rb')
Sy_val,y_val = pickle.load(pkl_val)

pkl_path_test = os.path.join(pkl_dir,"scattering_fold-test"+pkl_half+".pkl")
pkl_test = open(pkl_path_test,'rb')
Sy_test,y_test = pickle.load(pkl_test)

#log scale p and D
for idx in range(2,4):
    y_train[:,idx] = [math.log10(i) for i in y_train[:,idx]]
    y_test[:,idx] = [math.log10(i) for i in y_test[:,idx]]
    y_val[:,idx] = [math.log10(i) for i in y_val[:,idx]]

# normalization of the physical parameters
scaler = MinMaxScaler()
scaler.fit(y_train[:,:-2])
y_train_normalized = scaler.transform(y_train[:,:-2])
y_val_normalized = scaler.transform(y_val[:,:-2])
y_test_normalized = scaler.transform(y_test[:,:-2])

#log scale the input
#eps = 1e-11
Sy_train_log2 = np.asarray(np.log1p(((Sy_train>0)*Sy_train)/eps)).astype(np.float32)
Sy_val_log2 = np.asarray(np.log1p(((Sy_val>0)*Sy_val)/eps)).astype(np.float32)
Sy_test_log2 = np.asarray(np.log1p((Sy_test>0)*Sy_test/eps)).astype(np.float32)

y_train_u = np.asarray(2*(y_train[:,-2:]-0.5)).astype(np.float32)
y_train_theta = np.asarray(y_train_normalized).astype(np.float32)
y_val_u = np.asarray(2*(y_val[:,-2:]-0.5)).astype(np.float32)
y_val_theta = np.asarray(y_val_normalized).astype(np.float32)
y_test_u = np.asarray(2*(y_test[:,-2:]-0.5)).astype(np.float32)
y_test_theta = np.asarray(y_test_normalized).astype(np.float32)

#train the model


trial_dir = "../output/10trials_"+str(index)+"/tests"+str(trial)+"/"

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
test_loss = []
model_adjustable = create_model_adjustable(J=J,Q=Q,
                                            order=order,
                                            k_size=8,
                                            nchan_out=16,
                                            activation='linear',
                                            depth=depth, 
                                            units=units)
save_log = os.path.join(trial_dir,pickle_name+"_film_dep"+str(depth)+
    "_units_"+str(units)+"_score.pkl")
#model_adjustable.summary()
print('Start fitting the model...')
for epoch in range(30):
    np.random.shuffle(idx)
    Sy_temp = Sy_train_log2[idx[:m],:shape_time,:]
    y_temp_theta = y_train_theta[idx[:m],:]
    y_temp_u = y_train_u[idx[:m],:]

    hist = model_adjustable.fit([Sy_temp,y_temp_u],
                y_temp_theta,
                epochs=1,
                verbose=2,
                batch_size=bs,
                validation_data = ([Sy_val_log2[:,-shape_time:,:],y_val_u],y_val_theta),
                use_multiprocessing=False)

    validation_loss = hist.history['val_loss'][0]

    test_loss.append(model_adjustable.evaluate([Sy_test_log2,y_test_u],y_test_theta)[0])
    val_loss.append(validation_loss)
    train_loss.append(hist.history['loss'][0])

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        #epoch_str = "epoch-" + str(epoch).zfill(3)
        epoch_network_path = os.path.join(
           trial_dir, "_".join([ "J-" + str(J).zfill(2), 
            "Q-" + str(Q).zfill(2), "order" + str(order),
            "depth"+str(depth),
            "units"+str(units)]) + "_film.h5")
        model_adjustable.save(epoch_network_path)



        with open(save_log, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump([val_loss[-1],train_loss[-1],test_loss[-1]], filehandle)

print('Finished!')
















    
    
    