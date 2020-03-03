import pickle
import torch
from kymatio import Scattering1D
import os
import pandas as pd
import librosa
import numpy as np
import datetime

# scattering order one
def getsc_new(y,J,Q_num,order):
	"""
	this function outputs scattering transform of a time-domain signal.
	"""
	N = len(y)
	scattering = Scattering1D(J = J,shape=(N,), Q = Q_num, max_order=order)
	Sy = scattering(torch.Tensor(y))
	return Sy

def make_pickle(J,Q,order):
	df_val = pd.read_csv("./val_param.csv")
	val_id = df_val.values[:,0] 
	val_param = df_val.values[:,1:-1]
	J = 8
	Q = 1
	order = 2
	val_path = "/scratch/hh2263/drum_data/val/"
	num_val = len(val_id)
	for i,val in enumerate(val_id):
		filename = val_path + str(val) + "_sound.wav"
		x,fs = librosa.load(filename)
		Sy = getsc_new(x,J,Q,order).T #scattering coefficients
		Sy = Sy.reshape((Sy.shape[0],Sy.shape[1],1))
		y = val_param[i,:] #ground truth
		if i == 0:
			val_input = Sy
			val_gt = y
		else:
			val_input = np.dstack((val_input,Sy))
			val_gt = np.vstack((val_gt,y))


	#put all validation set into pickle file
	output = open('/scratch/hh2263/drum_data/val/'+'J_'+str(J)+'_Q_'+str(Q)+'_order_'+str(order)+'.pkl', 'wb')
	pickle.dump([val_input,val_gt], output)
	output.close()



if __name__ == '__main__':
	start_time = int(time.time())
	parser = argparse.ArgumentParser()

	parser.add_argument('J',type=int,default=8)
	parser.add_argument('Q',type=int,default=1)
	parser.add_argument('order',type=int,default=2)

	args = vars(parser.parse_args())

	print(str(datetime.datetime.now()) + " Start.")
	print("Generating validation pickle files.")
	##RUN THE FUNCTION
	make_pickle(**args)

	print(str(datetime.datetime.now()) + " Finish.")

	elapsed_time = time.time() - int(start_time)
	elapsed_hours = int(elapsed_time / (60 * 60))
	elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
	elapsed_seconds = elapsed_time % 60.
	elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(elapsed_hours,
	elapsed_minutes,
	elapsed_seconds)
	print("Total elapsed time: " + elapsed_str + ".")
