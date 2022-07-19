from kymatio.torch import Scattering1D
import librosa
import numpy as np
import torch
import sys
sys.path.append("../src")
import hcqt

#Make first order Time-Scattering 
def make_timesc_order1(J,N,Q,t,sample_ids,fold_str):
    timesc_order1 = Scattering1D(
            J = J, 
            shape = (N, ),
            Q = Q, 
            T=t, 
            max_order=1
            )

    X = []
    for sample_id in sample_ids:
        wav_name = str(sample_id) + "_sound.wav"
        wav_path = os.path.join(wav_dir, fold_str, wav_name)
        waveform, sr = sf.read(wav_path)
        torch_waveform = torch.Tensor(waveform)
        Sx = timesc_order1(torch_waveform).T
        X.append(Sx.detach().numpy())
    X = np.stack(X)
    return X


#Make second order Time-scattering
def make_timesc_order2(J,N,Q,t,sample_ids,fold_str):
    timesc_order2 = Scattering1D(
            J = J, 
            shape = (N, ),
            Q = Q, 
            T=t, 
            max_order=2
            )
    X = []
    for sample_id in sample_ids:
        wav_name = str(sample_id) + "_sound.wav"
        wav_path = os.path.join(wav_dir, fold_str, wav_name)
        waveform, sr = sf.read(wav_path)
        torch_waveform = torch.Tensor(waveform)
        Sx = timesc_order2(torch_waveform).T
        X.append(Sx.detach().numpy())
    X = np.stack(X)
    return X

#Make CQT
def make_cqt(waveform,b,sr,n_oct,fmin):
    #fmin default to 32.7Hz, covering the lowest frequency 40Hz
    Cx = librosa.cqt(waveform,sr=sr,n_bins=(n_oct)*b,hop_length=256,bins_per_octave=b) 
    return Cx

#Make HCQT
def make_hcqt(waveform,b,sr,n_oct,fmin):
    comp_hcqt = hcqt.HCQT(sr,bins_per_octave=b,harmonics=[0.5,1,2,3,4,5],f_min=32.7,
                 n_octaves=n_oct-2,hop_length=256) #choice of hop size?
    h_cqt = comp_hcqt.compute_hcqt(waveform,sr)
    return h_cqt

#Make VQT
def make_vqt(waveform,b,sr,n_oct,fmin):
    Vx = librosa.vqt(waveform,sr=sr,n_bins=(n_oct)*b,hop_length=256,bins_per_octave=b) 
    return Vx

