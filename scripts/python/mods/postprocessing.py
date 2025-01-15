import numpy as np
from scipy.signal import detrend

def variance(fields):
    n_times = fields.shape[0]
    mean = np.mean(fields, axis=0)
    return np.sum((fields - mean)**2, axis=0) / n_times

def autocorrelation(fields, lag=1):
    n_times = fields.shape[0]
    mean = np.mean(fields, axis=0)
    return np.sum((fields[:-lag] - mean) * (fields[lag:] - mean), axis=0) / n_times


def spatial_correlation(fields):
    detrended_fields = detrend(fields, axis=0, type='linear')
    detrended_fields = detrended_fields - np.mean(detrended_fields, axis=0)
    detrended_fields = np.pad(detrended_fields, ((0, 0), (1, 1), (1, 1)), mode='constant')
    
    fields = np.pad(fields, ((0, 0), (1, 1), (1, 1)), mode='constant')

    spatial_correlation = np.zeros(fields.shape[1:])
    for i in range(1, fields.shape[1] - 1):
        for j in range(1, fields.shape[2] - 1):
            spatial_correlation[i, j] = np.mean(fields[:, i, j] * fields[:, i+1, j+1])

    return spatial_correlation
    
    