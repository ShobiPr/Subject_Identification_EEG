#!/bin/python3.6
import statistics as stats
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import hilbert
import numpy as np
from pyhht import EMD
from scipy.spatial import distance
from filters import frequency_bands
import logging


# HHT-BASED FEATURES

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


# ------------------------------------------------------------------
# FRACTAL FEATURES

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

def get_energy_features(_vector, fs):
    feat = []
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

# EMD

def get_imfs_emd(signal):
    decomposer_signal = EMD(signal)
    return decomposer_signal.decompose()


def get_features_emd(instance, fs):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_emd(channel)
        if len(imfs) > 1: 
            features_vector += get_energy_features(imfs[:2], fs)
    return features_vector
