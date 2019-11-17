#!/bin/python3.6
import statistics as stats
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import hilbert
import numpy as np
from pyhht import EMD
from PyEMD import EEMD


# Extract IMFs from EEG
def get_imfs_emd(signal):
    try:
        # decomposer_signal = EMD(signal, fixe=100, n_imfs=2)
        decomposer_signal = EMD(signal, n_imfs=4)
        imfs = decomposer_signal.decompose()
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        return imfs[:2]
    except Exception as e:
        raise e


# ------------------------------------------------------------------
# HHT FEATURES

def instFreq(signal, fs):
    hs = hilbert(signal)
    omega_s = np.unwrap(np.angle(hs))
    return np.array(np.diff(omega_s) / (2 * np.pi / fs))


def instAmp(signal):
    hs = hilbert(signal)
    return np.abs(hs)


def marginal_frequency(signal, fs):
    return np.sum(instFreq(signal, fs))


def mean_instAmp(signal):
    i_a = instAmp(signal)
    return np.sum(i_a) / len(i_a)

# ------------------------------------------------------------------
# ENERGY FEATURES

def teager_energy(data):
    sum_values = sum(abs(data[x] ** 2) if x == 0
                     else abs(data[x] ** 2 - data[x - 1] * data[x + 1])
                     for x in range(0, len(data) - 1))
    return np.log10((1 / len(data)) * sum_values)


def instantaneous_energy(data):
    return np.log10((1 / len(data)) * sum(i ** 2 for i in data))


def pfd(a):
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


# ------------------------------------------------------------------

def get_statistics_values(imfs):
    feat = []
    # For each imf compute 9 values and return it in a single vector. (5 values in this example)
    # Mean, maximum, minimum, standard deviation, variance, kurtosis, skewness, sum and median
    for ii, imf in enumerate(imfs):
        feat += [
            # stats.mean(imf), #
            np.var(imf),
            np.std(imf),
            kurtosis(imf),
            # skew(imf), #
            np.max(imf),
            np.min(imf),
            # stats.median(imf) #
        ]
    return feat


def get_energy_values(imfs):
    feat = []
    # For each imf compute
    # for imf in imfs:
    for ii, imf in enumerate(imfs):
        feat += [
            instantaneous_energy(imf),
            teager_energy(imf),
            hfd(imf),
            pfd(imf)
        ]
    return feat


def get_HHT(imfs, fs):
    feat = []
    for i, imf in enumerate(imfs):
        feat += [
            marginal_frequency(imf, fs),
            mean_instAmp(imf)
        ]
    return feat


def get_values_f(_vector):
    feat = []
    fs = 200
    for ii, _vec in enumerate(_vector):
        feat += [
            instantaneous_energy(_vec),
            teager_energy(_vec),
            hfd(_vec),
            pfd(_vec),
            marginal_frequency(_vec, fs),
            mean_instAmp(_vec)
        ]
    return feat

# ------------------------------------------------------------------

def get_features_emd(instance, fs):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_emd(channel)
        features_vector += get_statistics_values(imfs)
        # features_vector += get_energy_values(imfs)
        # features_vector += get_HHT(imfs,fs)
        # features_vector += get_values_f(imfs)
    return features_vector



# ------------------------------------------

"""
def get_imfs_eemd1(signal):
    try:
        signal = np.array(signal)
        components = EEMD()(signal)
        imfs, res = components[:-1], components[-1]
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        return imfs[1:3]
    except Exception as e:
        print(e)
        return []


def get_imfs_EEMD(ch_freq_bands):
    ch_imfs = []
    for channel, bands in enumerate(ch_freq_bands):
        eIMFs = []
        for band, signal in enumerate(bands):
            eIMFs.append(get_imfs_eemd(signal))
        ch_imfs.append(eIMFs)
    return ch_imfs
"""