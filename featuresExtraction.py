#!/bin/python3.6
import numpy as np
import statistics as stats
from scipy.stats import kurtosis
from scipy.stats import skew
from pyhht import EMD
import matplotlib.pyplot as plt


def teager_energy(data):
    sum_values = sum(abs(data[x] ** 2) if x == 0
                     else abs(data[x] ** 2 - data[x - 1] * data[x + 1])
                     for x in range(0, len(data) - 1))
    return np.log10((1 / float(len(data))) * sum_values)


def instantaneous_energy(data):
    return np.log10((1 / float(len(data))) * sum(i ** 2 for i in data))





# Extract IMFs from EEG
def get_imfs(signal):
    try:
        signal = np.array(signal)
        decomposer_signal = EMD(signal, n_imfs=1)
        imfs = decomposer_signal.decompose()
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        # Return first IMF and residue
        return imfs[0:2]
    except Exception as e:
        print(e)
        return []




def first_order_diff(X):
    """ Compute the first order difference of a time series.
		For a time series X = [x(1), x(2), ... , x(N)], its	first order
		difference is:
		Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]
	"""
    D = []

    for i in range(1, len(X)):
        D.append(X[i] - X[i - 1])

    return D


def pfd(a):
    """Compute Petrosian Fractal Dimension of a time series """
    diff = np.diff(a)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]
    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(a)
    return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * N_delta)))


def hfd(a, k_max=None):
    L = []
    x = []
    N = a.size
    if not k_max:
        k_max = 10
    for k in range(1, k_max):
        Lk = 0
        for m in range(0, k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            Lmk = np.sum(np.abs(a[m + idxs * k] - a[m + k * (idxs - 1)]))
            Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
            Lk += Lmk
        L.append(float(np.log(Lk / (m + 1))))
        x.append([float(np.log(1 / k)), 1])
    (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
    return p[0]


# Extract statistics values from imfs
def get_statistics_values(imfs):
    feat = []
    # For each imf compute 9 values and return it in a single vector. (5 values in this example)
    # Mean, maximum, minimum, standard deviation, variance, kurtosis, skewness, sum and median
    for imf in imfs:
        # '''
        _mean = stats.mean(imf)
        _var = np.var(imf)
        _std = np.std(imf)
        _kurtosis = kurtosis(imf)
        _skew = skew(imf)
        _max = np.max(imf)
        _min = np.min(imf)
        _median = stats.median(imf)
        feat += [_mean, _var, _std, _kurtosis, _skew, _max, _min, _median]
    return feat


# Extract energy values from imfs
def get_energy_values(imfs):
    feat = []
    # For each imf compute
    for imf in imfs:
        _energy = instantaneous_energy(imf)
        _teager_energy = teager_energy(imf)
        _hfd = hfd(imf, 50)
        _pfd = pfd(imf)
        feat += [_energy, _teager_energy, _hfd, _pfd]
    return feat


def get_values_f(_vector):
    feat = []
    for ii, _vec in enumerate(_vector):
        feat += [
            # stats.mean(_vec),
            # np.var(_vec),
            np.std(_vec),
            # np.median(_vec),
            # min(_vec),
            # max(_vec),
            # sum(_vec),
            # skew(_vec),
            # kurtosis(_vec),
            instantaneous_energy(_vec),
            teager_energy(_vec),
            hfd(_vec),
            pfd(_vec),
        ]
    return feat


# Extract features for each channel
# Extract feature from each instance
# def get_features(channels):
def get_features(instance):
    features_vector = []
    # channels = channels[0:2]
    for i, channel in enumerate(instance):
        if i < 3:
            # Compute the imf for ech channel
            imfs = get_imfs(channel)
            # Compute feature values for imfs corresponding to one channel and join
            # features_vector += get_statistics_values(imfs)
            # features_vector += get_energy_values(imfs)
            features_vector += get_values_f(imfs)
    return features_vector


