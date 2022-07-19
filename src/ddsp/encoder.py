from ddsp.training import nn
import gin
import tensorflow.compat.v2 as tf

from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D,ReLU
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.constraints import nonneg



@gin.register
class wav2shapeEncoder(nn.DictLayer):
    """Predicts shapes of drum theta from scattering transform.
    """
    def __init__(self,
                 bins_per_oct,
                 activation,
                 lr,
                 input_keys=('feat'),
                 output_splits=(('y_predicted',5)),
                               name='wav2shape_encoder'):
        """Constructor."""
        super().__init__(name=name)
        
        self.input_keys = input_keys
        self.output_splits = output_splits
        
        momentum=0.5

        #model parameters
        self.bins_per_oct = bins_per_oct
        self.activation = activation

        #layers
        self.batchnorm_1 = BatchNormalization(momentum=momentum)
        self.batchnorm_2 = BatchNormalization(momentum=momentum)
        self.batchnorm_3 = BatchNormalization(momentum=momentum)
        self.batchnorm_4 = BatchNormalization(momentum=momentum)
        self.batchnorm_5 = BatchNormalization(momentum=momentum)
        self.batchnorm_6 = BatchNormalization(momentum=momentum)
        
        self.conv2d_1 = Conv2D(filters=128, kernel_size=(self.bins_per_oct,8), 
                             padding="same",kernel_regularizer=None)

        self.relu_activation = Activation("relu")
        self.avgpool2d_1 = AveragePooling2D(pool_size=(1,8),padding="valid")
        self.conv2d_2 = Conv2D(filters=64, kernel_size=(self.bins_per_oct,4),
                              padding="same",kernel_regularizer=None)
        self.conv2d_3 = Conv2D(filters=64, kernel_size=(self.bins_per_oct,4),
                              padding="same",kernel_regularizer=None)

        self.conv2d_4 = Conv2D(filters=8,kernel_size=(self.bins_per_oct,1),
                               padding="same",kernel_regularizer=None)


        self.flatten = Flatten()
        self.dense_mid = Dense(64,kernel_regularizer=None)

        self.dense_out = Dense(5, kernel_regularizer=None, kernel_constraint=nonneg(),bias_constraint=nonneg())
        self.act = Activation(self.activation)
        

    def call(self, feat)->['y_predicted']:
        """Converts features to (w11,tau11,p,D,alpha).

        Args:
          cqt features

        Returns:
          physical parameters

        """
        inputs = feat[...,None]  #take the scaled cqt coefficients, [bs,nfreq,ntime,nchan]
        #print("input feature",tf.reduce_max(inputs),tf.reduce_min(inputs))
        #S_input = keras.layers.Input(shape=self.S_input_shape)
        x = self.batchnorm_1(inputs)
        #print("after batchnorm",tf.reduce_max(x),tf.reduce_min(x),x.shape)
        x = self.conv2d_1(x)
        x = self.batchnorm_2(x)
        x = self.relu_activation(x)
        x = self.avgpool2d_1(x)
        #print("mark 1",tf.reduce_max(x),tf.reduce_min(x),x.shape)
        x = self.conv2d_2(x)
        x = self.batchnorm_3(x)
        x = self.relu_activation(x)
        #print("mark 2",tf.reduce_max(x),tf.reduce_min(x),x.shape)
        x = self.conv2d_3(x)
        x = self.batchnorm_4(x)
        x = self.relu_activation(x)
        x = self.avgpool2d_1(x)
        #print("mark 3",tf.reduce_max(x),tf.reduce_min(x),x.shape)
        x = self.conv2d_4(x)
        x = self.batchnorm_5(x)
        x = self.relu_activation(x)
        #print("mark 4",tf.reduce_max(x),tf.reduce_min(x),x.shape)
        x = self.flatten(x)
        x = self.dense_mid(x)
        #print("mark 5",tf.reduce_max(x),tf.reduce_min(x),tf.math.reduce_std(x),x.shape)
        #x = self.batchnorm_6(x) #why does this set everything to zero??
        #print("mark 6",tf.reduce_max(x),tf.reduce_min(x),x.shape,tf.math.reduce_std(x))
        
        x = self.relu_activation(x)
        #print("mark 6",tf.reduce_max(x),tf.reduce_min(x),x.shape,tf.math.reduce_std(x))
        
        x = self.dense_out(x)
        #print("after last dense",tf.reduce_max(x),tf.reduce_min(x),tf.math.reduce_std(x),x.shape)
        x = self.act(x)
        
        return {'y_predicted':x}
          
