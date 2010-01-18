import tensorflow as tf
import tensorflow.keras.backend as K



def mse_loss(y_true,y_pred):
    return K.mean(K.sum(K.square(y_pred-y_true),axis=-1))

def wrapper(M):
def mse_quadratic(y_true, y_pred):
    diff = y_true - y_pred #(bs,5)
    loss = tf.matmul(diff[:,None,:], tf.matmul(M, diff[:,:,None])) #(bs,1,5) (bs,5,5) (bs,5,1)
    return K.mean(loss)