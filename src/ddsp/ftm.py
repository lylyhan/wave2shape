from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf
import copy
import numpy as np

def tf_float64(x):
    """Ensure array/tensor is a float64 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float64)  # This is a no-op if x is float64.
    else:
        return tf.convert_to_tensor(x, tf.float64)

def tf_float32(x):
    """Ensure array/tensor is a float64 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float64.
    else:
        return tf.convert_to_tensor(x, tf.float32)
    

@gin.register
class FTM(processors.Processor):
    """Synthesize percussive sounds based on physical parameters."""

    def __init__(self,
                 n_samples=2**16,
                 sample_rate=22050,
                 mode=10,
                 input_keys=('y_predicted'),
                 name='ftm'):
        super().__init__(name=name) #use super to call processor's init class
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.mode = mode
  

   
    def getsounds_imp_linear_nonorm(self,m1,m2,x1,x2,h,tau11,w11,p,D,l0,alpha_side):
        """
        This implements Rabenstein's drum model. The inverse SLT operation is done at the end of each second-
        -order filter, no normalization on length and side length ratio is done
        """
        #print(w11,tau11,p,D,l0,alpha_side)
        w11 = tf_float64(w11)
        tau11 = tf_float64(tau11)
        p = tf_float64(p)
        D = tf_float64(D)
        l0 = tf_float64(l0)
        alpha_side = tf_float64(alpha_side)
        
        l2 = l0 * alpha_side
        s11 = -1 / tau11
        pi = tf_float64(np.pi)

        beta_side = alpha_side + 1 / alpha_side
        S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
        c_sq = (alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2 + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)) * (l0 / np.pi)**2
        T = c_sq 
        d1 = 2 * (1 - p * beta_side) / tau11
        d3 = -2 * p * alpha_side / tau11 * (l0 / pi)**2 

        EI = S**4 

        mu = tf_float64(tf.range(1,m1+1)) #(0,1,2,..,m-1)
        mu2 = tf_float64(tf.range(1,m2+1))
        dur = self.n_samples
        Ts = 1/self.sample_rate
        n = (mu[None,:] * pi / l0)**2 + (mu2[None,:] * pi / l2[:,None])**2 #eta 
        n2 = n**2 
        K = tf.sin(mu * pi * x1) * tf.sin(mu2 * pi * x2) #mu pi x / l (mode)

        beta = EI[:,None] * n2[None,:] + T[:,None] * n[None,:] #(bs,m)
        alpha = (d1[:,None] - d3[:,None] * n[None,:])/2 # nonlinear
        omega = tf.math.sqrt(tf.abs(beta - alpha**2))
        #print(beta.shape,alpha.shape,omega.shape,K.shape) #(1,bs,mode)
        N = l0 * l2 / 4
        yi = h * tf.sin(mu[None,:] * pi * x1) * tf.sin(mu2[None,:] * pi * x2) / omega[:,None] #(bs,mode)

        time_steps = tf.linspace(0,dur,dur) / self.sample_rate #(T,)
        y = tf.math.exp(-alpha[...,None] * time_steps[None,:]) * tf.sin(omega[...,None] * time_steps[None,:]) # (1,bs,mode,T)
        y = yi[...,None] * y
        y = tf.reduce_sum(y * K[None,None,:,None] / N[None,:,None,None],axis=-2) #impulse response itself
        #print("max val before scale",y.shape,tf.reduce_max(y,axis=-1)) #(1,1,bs,T)
        y = tf_float64(y) / tf.reduce_max(y,axis=-1)[...,None]
        #print("max val after scale",tf.reduce_max(y,axis=-1)) #(1,1,bs,T)
        return tf_float64(y)

    def get_controls(self,y_predicted):
        """Convert neural network output tensors into a dictionary of synthesizer controls.

        Args:
          theta: 2-D tensor of shape parameters [batch, 5]


        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        eps = 1e-5 #minimum tau_est
        #print("original prediction ",y_predicted)
        #if prediciting beta/alpha
        alpha_est = []
        for i in range(y_predicted.shape[0]):
            print("prediction",pow(10,-y_predicted.numpy()[i,-1]))
            print("just in case",np.roots([pow(10,-y_predicted.numpy()[i,-1]),-1,-1]),pow(10,-y_predicted.numpy()[i,-1]))
            root = max(np.roots([pow(10,-y_predicted.numpy()[i,-1]),-1,-1]))
            alpha = min(root,1/root)
            alpha_est.append(alpha)
        #print("after scaling alpha ",pow(10,y_predicted[:,0]),y_predicted[:,1],pow(10,-y_predicted[:,2]),
        #      pow(10,-y_predicted[:,3]),alpha_est)
        

        return {'w_est': pow(10,y_predicted[:,0]),
                'tau_est':y_predicted[:,1]+eps,
                'p_est': pow(10,-y_predicted[:,2]),
                'D_est': pow(10,-y_predicted[:,3]),
                'alpha_est':tf.convert_to_tensor(np.array(alpha_est),dtype=tf.float32)}

    def get_signal(self, w_est,tau_est,p_est,D_est,alpha_est):
        """Synthesize audio with additive synthesizer from controls.

        Args:
          mode: number of modes
          w_est: fundamental frequency in hz
          tau_est: float 0.1~3 not in seconds
          p: float, roundness
          D: float, inharmonicity
          alpha: float between 0 and 1, ratio between width and height of the drum
          sample_rate: samples/second


        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        signal = self.getsounds_imp_linear_nonorm(m1=self.mode,
                                             m2=self.mode,
                                             x1=0.5,
                                             x2=0.5,
                                             h=0.03,
                                             tau11=tau_est,
                                             w11=w_est,
                                             p=p_est,
                                             D=D_est,
                                             l0=np.pi,
                                             alpha_side=alpha_est)

        return signal


