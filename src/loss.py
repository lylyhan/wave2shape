import tensorflow as tf
import tensorflow.keras.backend as K



def mse_loss(y_true,y_pred):
    return K.mean(K.sum(K.square(y_pred-y_true),axis=-1))