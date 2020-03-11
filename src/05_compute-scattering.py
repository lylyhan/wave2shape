from kymatio import Scattering1D
import librosa
import numpy as np
import os
import pandas as pd
import pickle
import torch
import tqdm

# Parse input arguments.
args = sys.argv[1:]
fold_str = args[0]
J = int(args[1])
order = int(args[2])
