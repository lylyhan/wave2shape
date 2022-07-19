import ddsp
from ddsp.training.data import DataProvider
import gin
import tensorflow.compat.v2 as tf
import pandas as pd
import soundfile as sf
import pescador
import numpy as np


# ------------------------------------------------------------------------------
# FTM Data for wav2shape
# ------------------------------------------------------------------------------
@gin.register
class FTMProvider(DataProvider):
    """
    Read from customized FTM dataset
    """
    def __init__(self, split, data_dir, param_dir, sample_rate, frame_rate=1000):
        """TfdsProvider constructor.

            Args:
            name: TFDS dataset name (with optional config and version).
            split: Dataset split to use of the TFDS dataset.
            data_dir: The directory to audio files in drum dataset.
            param_dir: The directory to the csv file containing ground truth parameters
            sample_rate: Sample rate of audio in the dataset.
            frame_rate: Frame rate of features in the dataset.
        """
        #self._name = name
        self._split = split
        self._data_dir = data_dir
        self._param_dir = param_dir
        self.N = 2**16 #signal length
        self.df = pd.read_csv(self._param_dir)
        
        super().__init__(sample_rate, frame_rate)


    
    @pescador.streamable
    def feature_sampler(ids,fold,params_normalized,idx,audio_path):
        """
        output a {input, ground truth} pair for the designated audio sample
        """
        i = idx
        y = params_normalized[i,:] #ground truth
        #load weights here!
        fullpath = os.path.join(audio_path,fold,str(ids[i])+"_sound.wav") 
        x,sr = sf.read(fullpath)
        while True:
            yield {'input': np.array(x)}#,'y': y} 

    def data_generator(ids,fold,params_normalized, batch_size, idx, active_streamers,
                        rate,random_state=12345678):
        """
        use streamers to output a batch of {input groundtruth} pairs. 
        """
        #load the weights npy file
        streams = [feature_sampler(ids,fold,params_normalized,i,audio_path) for i in idx]
        # Randomly shuffle the eds
        random.shuffle(streams)
        mux = pescador.StochasticMux(streams, active_streamers, rate=rate, random_state=random_state)
        return pescador.maps.buffer_stream(mux, batch_size)

   #generator
    def ftm_generator(self):
        sample_ids = self.df.values[:, 0]
        idx = np.arange(0,sample_ids.shape[0],1)
        np.random.shuffle(idx)
        #print(idx[0])
        
        params = self.df.values[idx[0], 1:-1]
        #yield a batch of audio examples
        filename = os.path.join(self._data_dir,self._split,
                                str(self.df.values[idx[0],0])+"_sound.wav")      
        y, sr = sf.read(filename)
        yield np.array(y).astype(np.float64) #,np.array(params).astype(np.float64)

    def get_dataset(self, shuffle=True):
        """Read dataset.

        Args:
          shuffle: Whether to shuffle the input files.

        Returns:
          dataset: A tf.data.Dataset that reads from TFDS.
        """

        return tf.data.Dataset.from_generator(self.ftm_generator,
                                              output_signature=(tf.TensorSpec(shape=(self.N,), dtype=tf.float64)))