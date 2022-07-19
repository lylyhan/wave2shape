import ddsp
from ddsp.training import preprocessing,nn
import gin
import tensorflow.compat.v2 as tf
tf.config.run_functions_eagerly(True)
from kymatio.tensorflow import Scattering1D
import numpy as np
import librosa



@gin.register
class ScatteringPreprocessor(nn.DictLayer):
    """Class that calculates scattering coefficients of a raw waveform"""

    def __init__(self, scattering=None,eps=1e-3):
        super().__init__() 
        self.eps = eps
        self.scattering = scattering
        if not self.scattering: 
            raise ValueError("scattering shouldn't be none")


    def __call__(self, features, training=True):
        features = super().__call__(features, training)
        return self._scattering_processing(features)

    def _scattering_processing(self, features):
        """generate 1D scattering coefficients of input raw audio (features)"""  

        audio = features['raw_audio'] #but where does the audio come from?
        features['scattering'] = self.scattering(audio)
        features['scattering_scaled'] = tf.log1p(((features['scattering']>0)*features['scattering'])/self.eps)
        return features




@gin.register
class CQTPreprocessor(nn.DictLayer):
    """Class that calculates CQT coefficients of a raw waveform"""

    def __init__(self,Q,sr,n_oct,fmin,eps=1e-3,output_keys=['feat'],
              **kwargs):
        super().__init__(output_keys=output_keys,**kwargs) 
        self.eps = eps
        self.Q = Q
        self.sr = sr
        self.n_oct = n_oct
        self.fmin = fmin
        self.eps = eps

    def __call__(self,audio=None)->['feat']:
        if type(audio) is dict:
            audio = audio['audio']
        cqt = librosa.cqt(audio.numpy(),sr=self.sr,n_bins=(self.n_oct)*self.Q,hop_length=256,bins_per_octave=self.Q) 
        if self.eps:
            cqt = np.log1p(cqt/self.eps)
        #print(np.max(cqt),np.min(cqt))
        return {'feat':tf.convert_to_tensor(cqt,dtype=tf.float64)} #how did i deal with complex number in ploss model?
