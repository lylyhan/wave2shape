import librosa
import numpy as np



class HCQT():
	def __init__(
		self,
		sr,
		bins_per_octave,
		harmonics,
		n_octaves,
		f_min,
		hop_length):

		self.sr = sr
		self.bins_per_octave = bins_per_octave
		self.harmonics = harmonics
		self.n_octaves = n_octaves
		self.f_min = f_min
		self.hop_length = hop_length



	def compute_hcqt(self,y,fs):
	    cqt_list = []
	    shapes = []
	    for h in self.harmonics:
	        #print(h)
	        cqt = librosa.cqt(
	            y, sr=fs, hop_length=self.hop_length, fmin=self.f_min*float(h),
	            n_bins=self.bins_per_octave*self.n_octaves,
	            bins_per_octave=self.bins_per_octave
	        )
	        cqt_list.append(cqt)
	        shapes.append(cqt.shape)
	    
	    shapes_equal = [s == shapes[0] for s in shapes]
	    if not all(shapes_equal):
	        min_time = np.min([s[1] for s in shapes])
	        new_cqt_list = []
	        for i, cqt in enumerate(cqt_list):
	            new_cqt_list.append(cqt[:, :min_time])
	            cqt_list.pop(i)
	        cqt_list = new_cqt_list

	    log_hcqt = 20.0*np.log10(np.abs(np.array(cqt_list)) + 0.0001)
	    log_hcqt = log_hcqt - np.min(log_hcqt)
	    log_hcqt = log_hcqt / np.max(log_hcqt)
	    return log_hcqt