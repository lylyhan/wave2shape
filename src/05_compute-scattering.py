from kymatio import Scattering1D
import librosa
import numpy as np
import os
import pandas as pd
import pickle
import torch
import tqdm

# Define output path.
output_dir = "/scratch/vl1019/drum_data/"
os.makedirs(output_dir, exist_ok=True)


# Parse input arguments.
args = sys.argv[1:]
fold_str = args[0]
J = int(args[1])
order = int(args[2])
Q = 1


# Start counting time.
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print("Computing scattering features.")


# Load CSV file of physical parameters.
wav_dir = os.path.join("/scratch/hh2263/drum_data", fold_str)
csv_path = "../notebooks/" + fold_str + "_param.csv"
df = pd.read_csv(csv_path)
sample_ids = df_tr.values[:, 0]


# Define scattering operator
N = 2**16
scattering = Scattering1D(
    J=J,shape=(N,), Q=Q, max_order=order)


# Compute scattering features.
X = []
for sample_id in tqdm.tqdm(sample_ids):
    wav_name = str(sample_id) + "_sound.wav"
    wav_path = os.path.join(wav_dir, wav_name)
    waveform, _ = librosa.load(wav_path)
    torch_waveform = torch.Tensor(waveform)
    Sx = np.array(scattering(torch_waveform).T)
    X.append(Sx)
X = np.stack(X)


# Load physical parameters. (ground truth)
y = df.values[:,1:-1]


# Export to pickle file.
fold_dir = os.path.join(output_dir, fold_str)
os.makedirs(fold_dir, exist_ok=True)
pickle_name = "_".join([
    "scattering",
    "J-" + str(J).zfill(2),
    "Q-" + str(Q).zfill(2),
    "order" + str(order)
])
pickle_path = os.path.join(fold_dir, pickle_name + ".pkl")
with open(pickle_path, 'wb') as pickle_file:
    pickle.dump([X, y], pickle_file)


# Print elapsed time.
print(str(datetime.datetime.now()) + " Finish.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
    elapsed_hours, elapsed_minutes, elapsed_seconds)
print("Total elapsed time: " + elapsed_str + ".")
