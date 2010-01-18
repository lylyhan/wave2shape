import datetime
import pandas as pd
import soundfile as sf
from kymatio.torch import Scattering1D
import librosa
import numpy as np
import os
import pickle
import time
import torch
import tqdm
import sys
sys.path.append("../src")
import hcqt
import features

# Define output path.
data_dir = "/home/han/data/drum_data/"
pickle_dir = os.path.join(data_dir, "han2022features-pkl")
wav_dir = data_dir
os.makedirs(pickle_dir, exist_ok=True)


# Parse input arguments.
#args = sys.argv[1:]
#fold_str = args[0] #train, val, test
#J = int(args[1]) #12, 14
#Q = int(args[2]) #16
N = 2**16
sr = 22050

# Set hyperparameters for all
#n_oct = J
t = 2**11
#b = Q
#fmin = 0.4*sr*2**(-J)




if __name__ == "__main__":

    # Start counting time.
    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print("Computing scattering features.")
    for fold_str in ["train","test","val"]:
        # Load CSV file of physical parameters.
        csv_path = os.path.join(data_dir,"annotations",fold_str + "_param_v2.csv")
        df = pd.read_csv(csv_path)
        sample_ids = df.values[:, 0]
        # Load physical parameters. (ground truth)
        y = df.values[:, 1:-1]

        for i,feature in enumerate(['scattering_o1','scattering_o2']):
            if i == 0:
                Js = [8,10] #different j for different orders
            else:
                Js = [14]

            for J in Js:
                for Q in [1,16]:
                    # Export to pickle file.
                    os.makedirs(os.path.join(pickle_dir,feature),exist_ok=True)
                    pickle_name = "_".join(
                        [feature,
                         "fold-"+str(fold_str),"J-" + str(J).zfill(2), "Q-" + str(Q).zfill(2)]
                    )
                    pickle_path = os.path.join(pickle_dir, feature,pickle_name + ".pkl")
                    #if it doesn't already exist
                    if not os.path.exists(pickle_path) and not os.path.exists(pickle_path[:-4]):
                        print("making features for ",pickle_path)
                        if i==0:
                            X = features.make_timesc_order1(J,N,Q,t,sample_ids,fold_str)
                            #np.save(pickle_path,[X, y],allow_pickle=True)
                            with open(pickle_path, 'wb') as pickle_file:
                                pickle.dump([X, y], pickle_file)

                        elif i==1:
                            #make files separately
                            pickle_folder = os.path.join(pickle_dir, feature, pickle_name)
                            os.makedirs(pickle_folder, exist_ok=True)
                            features.make_timesc_order2(J,N,Q,t,sample_ids,fold_str,[pickle_folder,pickle_name],y) 
                            #this needs to be dumped separately
                        print("finished saving for", pickle_path)





    # Print elapsed time.
    print(str(datetime.datetime.now()) + " Finish.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
        elapsed_hours, elapsed_minutes, elapsed_seconds
    )
    print("Total elapsed time: " + elapsed_str + ".")

