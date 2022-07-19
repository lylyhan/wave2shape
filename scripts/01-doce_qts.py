import doce
import time
import numpy as np
from pathlib import Path
import sys
sys.path.append("../src")
import train

# invoke the command line management of the doce package
if __name__ == "__main__":
    doce.cli.main()
    
def mean_min(data): # average over the minimal values of each row
    return np.mean(np.min(data, axis = 1))

# define the doce environnment
def set(args=None):
    # define the experiment
    experiment = doce.Experiment(
    name = 'wav2shape_multitask_qts',
    purpose = 'multitask regression experiments with cqt variations',
    author = 'han han',
    address = 'han.han@ls2n.fr',
    )

    # set acces paths (here only storage is needed)
    experiment.set_path('output', '~/wave2shape/doce_out/'+experiment.name+'/')

    #experiment setup
    experiment.batchsize = 64
    experiment.n_epoch = 30
    experiment.steps_per_epoch = 50
    experiment.is_multitask = True
    experiment.is_normalize = True
    experiment.logscale = 1e-3
    experiment.n_trials = 8
    
    # set the plan (factor : modalities) #how to set modalities that only apply to certain feature again?
    experiment.add_plan('plan',
                       feature_type = ['cqt','hcqt','vqt'],
                       J = [6,8],
                       Q = [8,16,24],
                       #denselayer = ['dense','deep_centroid'],  #omegaonly
                       activation_type = ['linear'],                    
    )
    
    # set the metrics
    experiment.set_metrics(
    training_loss = ['0', 'min', '-1','mean_min'], #variable name needs to match the npy file name, perform statistics on the saved matrix
    validation_loss = ['min', '-1','mean_min'],
    #everything = [] #customized function 
    )
    return experiment

def step(setting, experiment):
    print(setting.identifier())
    # this is where experiments are called
    ftype = setting.feature_type
    J = setting.J
    Q = setting.Q
    activation = setting.activation_type
    batchsize = experiment.batchsize
    n_epoch = experiment.n_epoch
    steps_per_epoch = experiment.steps_per_epoch
    is_normalize = experiment.is_normalize
    is_multitask = experiment.is_multitask
    logscale = experiment.logscale
    n_trials = experiment.n_trials
    
    if activation == "sigmoid":
        lr = 0.001
    else:
        lr = 0.01
        
    # a function that outputs loss and saves model chekcpoint somewhere else
    validation_loss = []
    training_loss = []
    for trial in range(n_trials):
        val_loss,train_loss = train.run_train({"type":ftype,"J":J,"Q":Q},
                                                        trial,
                                                        logscale=logscale,
                                                        is_normalize=is_normalize,
                                                        is_multitask=is_multitask,
                                                        activation=activation,
                                                        loss="ploss",
                                                        batch_size=batchsize,
                                                        n_epoch=n_epoch,
                                                        lr=lr,
                                                        steps_per_epoch=steps_per_epoch)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        
    training_loss = np.stack(training_loss)
    validation_loss = np.stack(validation_loss)

    # storage of outputs (the string between _ and .npy must be the name of the metric defined in the set function)
    np.save(experiment.path.output+setting.identifier()+'_training_loss.npy', training_loss)
    np.save(experiment.path.output+setting.identifier()+'_validation_loss.npy', validation_loss)
  