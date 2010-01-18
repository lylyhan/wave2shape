import math
import numpy as np
import os

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
    
def welford(feature_path):
    """
    compute running variance of a collection of feature vectors
    """
    count = 1
    files = [os.listdir(os.path.join(feature_path,i)) for i in os.listdir(feature_path) if os.path.isdir(os.path.join(feature_path,i))]
    feat_files = sum(files,[])
    mean = np.load(os.path.join(feature_path, feat_files[0].split("_")[0], feat_files[0])).squeeze()
    var = np.zeros(mean.shape)
    existingAggregate = (count, mean, var)
    for i in range(1,len(feat_files)):
        newValue = np.load(os.path.join(feature_path, feat_files[i].split("_")[0],
                                        feat_files[i])).squeeze()
        existingAggregate = update(existingAggregate, newValue)
        mean_running,variance_running, sampvar_running = finalize(existingAggregate)
    return mean_running,variance_running,sampvar_running

def welford_stand(feature_path,mu,std):
    """
    compute running variance of a collection of feature vectors while performing standardization
    """
    count = 1
    feat_files = os.listdir(feature_path)
    mean = np.load(os.path.join(feature_path,feat_files[0])).squeeze()
    mean = (mean - mu)/std
    var = np.zeros(mean.shape)
    existingAggregate = (count, mean, var)
    for i in range(1,len(feat_files)):
        newValue = np.load(os.path.join(feature_path,feat_files[i])).squeeze()
        newValue = (newValue - mu) / std
        existingAggregate = update(existingAggregate, newValue)
        mean_running,variance_running, sampvar_running = finalize(existingAggregate)
    return mean_running,variance_running,sampvar_running