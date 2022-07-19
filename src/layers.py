import tensorflow.keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D,AveragePooling2D, Conv2D, MaxPooling2D,ReLU
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model #save and load models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint


class CentroidDense(tf.keras.layers.Layer):
    #the only difference with dense is that it has an extra step of flattening the input
    def __init__(self,units,
                 time_average=True,
                 activation=None,use_bias=True,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 bias_initializer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CentroidDense, self).__init__(**kwargs)
        self.units = units
        self.time_average = time_average
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape): 
        
        #not flattened input, (f,t,#filters)
        _,self.freq,self.time,self.width = input_shape #(f,t,#filters)
        if self.time_average:
            kern_shape = [self.width,self.units]
        else:
            kern_shape = [self.width*self.time, self.units]
            
        self.kernel = self.add_weight(
                        'kernel', 
                        shape=kern_shape, #if not pooling over time then self.width*self.time
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        dtype=self.dtype,
                        trainable=True)
        #initialize trainable weights
        if self.use_bias:
            self.bias = self.add_weight(
              'bias',
              shape=[self.units,],
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
            self.bias = None
    
        super(CentroidDense, self).build(input_shape)

    def call(self,inputs,vector,i): #the vector always match first dimension of input
        """
        i: along which channel of the inputs should vector be multiplied 
        """
        vector = K.constant(vector)
        print(inputs.shape,vector.shape)
        assert inputs.shape[i+1]==vector.shape[0] #inputs(bs, f, t, #filters), vectors (f)
        
        if self.time_average:
            #sum over time axis (whatever axis that is left out)
            #inputs = tf.reduce_sum(inputs,axis=i+2)  # (bs,f,#filters) # tf.mean instead
            inputs = tf.reduce_mean(inputs,axis=i+2)
            #normalize input into a pdf distribution, each filter should be normalized separately
            inputs = inputs/tf.reduce_sum(inputs,axis=[0,1],keepdims=True)#the rest of the two dimensions
            #multiply input with vector along frequency dimension/ or some defined dimension
            inputs = inputs*vector[None,:,None] 
        else:
            inputs = inputs/tf.reduce_sum(inputs,axis=[0,1,2],keepdims=True) #(bs,f,t,#filters)
            inputs = inputs*vector[None,:,None,None]
        
        print(inputs.shape)
        #sum over frequency dimension
        inputs = tf.reduce_sum(inputs,axis=1,keepdims=True) #there's an extra batchsize dimension (bs,)
        #flatten input
        print("after summing",inputs.shape)
        inputs = tf.reshape(inputs,[-1,tf.reduce_prod(inputs.shape[1:])]) #(bs, #filters)
        print("afterflatten",inputs.shape,self.kernel.shape)
        #multiply with kernel
        rank = inputs.shape.rank
        
        outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]]) #multiply kernel  (#in filters,#out units)
        print(outputs.shape,inputs.shape,self.bias.shape) 

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs[None,:], self.bias)
        return outputs

    def compute_output_shape(self, input_shape):

        return input_shape[:-1]

    def get_config(self):
        config = super(CentroidDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config
    



