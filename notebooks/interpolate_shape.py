import hitdifferentparts
from torch.autograd import backward
import numpy as np
import torch
from kymatio import Scattering1D
from sklearn.preprocessing import MinMaxScaler

def getsc(y, J):
	"""
	this function outputs scattering transform of a time-domain signal.
	"""
	N = len(y)
	scattering = Scattering1D(J = J,shape=(N,))
	Sy = scattering(y)
	return Sy


def interpolate_sounds(x_coord,y_coord,num,prec,J,w,tau,p,D,alpha):
		"""
		x_coord, y_coord is the point at which you want to interpolate the sounds
		num is number of interpolation points
		prec is distance between interpolated to interpolation points
		"""
		scs = []
		sounds = []
		if num == 2:
			inter = [(x_coord+prec,y_coord),(x_coord-prec,y_coord)]
		elif num == 4:
			inter = [(x_coord+prec,y_coord),(x_coord-prec,y_coord),(x_coord,y_coord+prec),(x_coord,y_coord-prec)]
		for (x,y) in inter:
			y = hitdifferentparts.hitdiffparts(x,y,w,tau,p,D,alpha)
			Sy = getsc(torch.Tensor(y),J)
			sounds.append(y)
			scs.append(Sy)
		return sum(scs)/float(num),sounds



def regress_signal(
		s_c, N, J,
		learning_rate=0.1,
		momentum=0.0,
		NAG = False,
		bold_driver_accelerator=1.1,
		bold_driver_brake=0.5,
		n_iterations=100,
		cost = "mse",
		verbose=False):
	"""
	this function finds time-domain signal from scattering transform
	input: scattering tranform tensor, length of the signal(implicitly), learning rate and number of iterations
	output: time-domain signal represented in a tensor
	"""
	scattering = Scattering1D(J=J,shape=(N,)) #N is length of signal, J is number of scales
	if cost == "cross-entropy":
		loss = torch.nn.KLDivLoss(reduction='batchmean')

	#random guess
	x = torch.randn((N,),requires_grad=True)
	Sx = scattering(x)
	#target
	Sy = s_c
	history = []
	signal_update = 0*x #

	#normalize Sy
	#Sy_normalized,Sy_sum,min_Sy,range_Sy = normalize(Sy) # Sy is the target
	Sy_normalized = normalize(Sy)

	#iterate to regress random guess to be close to target
	for it_id in range(n_iterations):
		# Backpropagation
		if cost == "mse":
			err = torch.norm(Sx-Sy)
			#backward(torch.norm(Sx - Sy))
		elif cost == "cross-entropy":
			#Sy = Sy.long()
			#np.sum(yHat * np.log((yHat / y)))
			#normalize Sx and Sy - tensor values should always be 0-1 due to nature of cross entropy (correlation between probabilities)
			#Sx_normalized,Sx_sum,min_Sx,range_Sx = normalize(Sx) 
			Sx_normalized = normalize(Sx)
			err = torch.sum(Sx_normalized * torch.log((Sx_normalized / Sy_normalized)))
			#err = loss(torch.log(Sx_normalized), torch.log(Sy_normalized))
			print(err,torch.norm(Sx_normalized,p=2),torch.norm(Sy_normalized,p=2))
			#backward(loss(Sx, Sy))         
		backward(err)
		if NAG == False:
			delta_x = x.grad 

		else:
			# for some reasons this nag doesn't have gradient!
			nag_x = torch.tensor(momentum * signal_update + x, requires_grad = True)
			print(nag_x.grad,nag_x)
			delta_x = nag_x.grad
			print(delta_x)

			#print(x.grad)

			# Gradient descent
		with torch.no_grad():
			momentum = min(0.99,1-2**(-1-np.log(np.floor(it_id/250)+1)/np.log(2))) #according to paper, adaptive momentum
			signal_update = momentum * signal_update - delta_x * learning_rate
			new_x = x + signal_update
		new_x.requires_grad = True

		# New forward propagation
		Sx = scattering(new_x)
		# Measure the new loss
		history.append(err)

		if history[it_id]> history[it_id-1]:
			learning_rate *= bold_driver_brake
		else:
			learning_rate *= bold_driver_accelerator
			x = new_x

	return x,history


def getscaler():
	df_train = pd.read_csv("../notebooks/train_param.csv")
	params = df_train.values[:,1:-1]
	for idx in range(2,4):
		params[:,idx] = [math.log10(i) for i in params[:,idx]]

	scaler = MinMaxScaler()
	scaler.fit(params)

	return scaler

def create_model_adjustable(J,Q,order,k_size,nchan_out,activation):
	N = 2**15
	y = np.random.rand(N)
	scattering = Scattering1D(J = J,shape=(N,), Q = Q, max_order=order)
	Sy = np.array(scattering(torch.Tensor(y))).T
	input_x,input_y = Sy.shape
	nchan_in = 1       # number of input channels.  1 since it is BW

	input_shape = (input_x,input_y)#Sy.shape
	kernel_size = (k_size,)
	K.clear_session()
	model=Sequential()
	#1 conv layer +  1 batch normalization + nonlinear activation + pooling
	model.add(BatchNormalization(input_shape=input_shape))
	model.add(Conv1D(filters=nchan_out,
		kernel_size=kernel_size, padding="same",name='conv1'))
	#model.add(BatchNormalization())
	model.add(Activation("relu"))

	if model.layers[-1].output_shape[1]>=4:
		pool = 4
	elif model.layers[-1].output_shape[1]==2:
		pool = 2
	    
	model.add(AveragePooling1D(pool_size=(pool,)))


	for i in range(3):
		model.add(Conv1D(filters=nchan_out,
		             kernel_size=kernel_size, padding="same" ))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		#print('before pool',model.layers[-1].output_shape)
		if model.layers[-1].output_shape[1] >= 4:
			model.add(AveragePooling1D(pool_size=(4,)))
		elif model.layers[-1].output_shape[1] == 2:
			model.add(AveragePooling1D(pool_size=(2,)))
		#print(model.layers[-1].output_shape)

	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	#what activation should be chosen for last layer, for regression problem? should be a linear function
	model.add(Dense(5, activation=activation)) #output layer that corresponds to the 5 physical parameters.


	# Compile the model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])




	return model

def interpolate_shape(x,y,num,prec,J,model_path,w,tau,p,D,alpha):
    #prepare scattering features 
	Sy_interpolated,sounds = interpolate_sounds(x,y,num,prec,J,w,tau,p,D,alpha)#this J should be small to recover sound	
	Sy_interpolated_log = np.log1p(((Sy_interpolated>0)*Sy_interpolated)/1e-11)
	Sy_interpolated_log = Sy_interpolated_log.T
	n,m = Sy_interpolated_log.shape

	#prepare normalized ground truth
	scaler = getscaler()
	gt_original = np.array([w,tau,p,D,alpha]).reshape((1,5))
	gt_normalized = scaler.transform(gt_original)

	#prepare model
	model_best = create_model_adjustable(J,1,2,k_size=8,nchan_out=16,activation='linear')
	model_best.load_weights(model_path)
	hist = model_best.evaluate(np.array(Sy_interpolated_log).reshape(1,n,m),gt_normalized)
	model_predict = model_best.predict(Sy_interpolated_log.reshape(1,n,m))
	return hist,model_predict








