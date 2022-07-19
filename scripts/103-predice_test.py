import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append("../src")
sys.path.append("/home/han/kymatio-jtfs/")
sys.path.append("../scripts")

import train
import features
import data_generator
import cnn

import pandas as pd
import librosa
import pescador
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(tf.test.is_gpu_available())

args = sys.argv[1:]
param = args[0] #beta_alpha or alpha
loss = args[1] #weighted_p or ploss




feat_path = "/home/han/data/drum_data/han2022features-pkl/jtfs_j14_q16/"
model_path = "/home/han/wave2shape/output/doce"
audio_path = "/home/han/data/drum_data/"



is_multitask = True
activation = "linear"
logscale = 0.001
batch_size = 64

N = 2**16
sr = 22050
ftype = "cqt"
J = 8
Q = 16


def find_best_trial(exp):
    min_vals = []
    for trial in os.listdir(exp):
        loss = pd.read_csv(os.path.join(exp,trial,"training.log"))
        min_vals.append(min(loss["val_loss"]))
    min_min = min(min_vals)
    return os.listdir(exp)[min_vals.index(min_min)]


def yield_prediction(feat_test,ckpt_path,model,is_generator):
    model.load_weights(ckpt_path)
    if is_generator:
        feat_test = pescador.tuples(test_batches,'input','y')
    preds = model.predict(feat_test,verbose=1)
    return preds

def make_model_predict(J,Q,y_test_normalized,loss,param,is_generator,feat_test=None):
    N = 2**16
    sr = 22050
    logscale = 0.001
    exp_type="_".join(["multitask"+str(True), loss, param, "linear", "log"+str(logscale)])
    exp_name = "_".join([ftype,"J"+str(J),"Q"+str(Q),"bs"+str(64),exp_type])
    trial_name = find_best_trial(os.path.join(model_path,exp_name))
    print(exp_name,trial_name)
    #figure out feature dimension
    waveform = np.zeros((N,))
    ex_input = features.make_cqt(waveform,Q,sr,J,fmin=32.7)
    ex_input = ex_input[:,:,None]
    #make model
    model = cnn.create_model_conv2d(bins_per_oct=Q,S=ex_input[None,:],activation="linear",is_multitask=True,lr=0.01)
    ckpt_path = os.path.join(model_path,exp_name,trial_name,"ckpt")
    #make test set data generator
    test_idx = np.arange(0,y_test_normalized.shape[0],1)
    #normalize test set ground truth
    if is_generator:
        feature_type = {"type":"cqt","J":J,"Q":Q}
        test_batches = data_generator.data_generator(test_ids,
                                               "test",
                                                y_test_normalized, 
                                                feature_type,
                                                audio_path,
                                                batch_size=64, 
                                                idx=test_idx,
                                                active_streamers=32,
                                                rate=64,
                                                random_state=44000,
                                                loss=loss,
                                                 param=param,
                                                eps=logscale)
    else:
        test_batches = feat_test
    #load weights to model and yield prediction
    preds = yield_prediction(test_batches,ckpt_path,model,is_generator)
    del model
    del feat_test
    tf.keras.backend.clear_session()
    return preds, os.path.join(model_path,exp_name,trial_name)


if __name__ == "__main__":
    #load test set ground truth
    y_test, test_ids = train.load_gt("test", param)
    y_train, train_ids = train.load_gt("train", param)
    y_val, val_ids = train.load_gt("val", param)

    y_train_normalized, y_val_normalized, y_test_normalized = train.preprocess_gt(y_train,y_val,y_test,param) 
    print("finished processing ground truth")
    # make model and predict on test set
    feat_cqt = np.load(os.path.join("/home/han/data/drum_data/han2022features-pkl/","CQT_J8_Q16_testset_features.npy"))
    if logscale:
        feat_cqt = np.log1p(feat_cqt/logscale)
    print("finished loading features, predicting")
    y_preds, model_path = make_model_predict(J,Q,y_test_normalized,loss,param,is_generator=False,feat_test=feat_cqt)
    
    pred_name = model_path + "_test_preds.npy"
    np.save(pred_name, y_preds)
    
    