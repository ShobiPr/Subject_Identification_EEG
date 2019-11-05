#!/bin/python3.6
import numpy as np
import statistics as stats
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import hilbert
from pyhht import EMD
from PyEMD import EEMD
from math import *
from decimal import Decimal
from hilbert_transform import hilbert_transform
from esemble_emd import get_imfs
from pyhht.utils import inst_freq
import matplotlib.pyplot as plt


def teager_energy(signal):
    data = hilbert(signal)
    sum_values = sum(abs(data[x] ** 2) if x == 0
                     else abs(data[x] ** 2 - data[x - 1] * data[x + 1])
                     for x in range(0, len(data) - 1))
    return np.log10((1 / float(len(data))) * sum_values)


def instantaneous_energy(signal):
    data = hs = hilbert(signal)
    return np.real(np.log10((1 / float(len(data))) * sum(i ** 2 for i in data)))


# Extract IMFs from EEG
def get_imfs_emd(signal):
    try:
        signal = np.array(signal)
        decomposer_signal = EMD(signal, n_imfs=5)
        imfs = decomposer_signal.decompose()
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        # Return first IMF and residue
        #return imfs[1:3]
        return imfs[1:3]
    except Exception as e:
        print(e)
        return []


def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 3)

# x = signal
# y = extracted IMF
def get_Minkowski_distance(x, y, p_value):
    return p_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)


def get_imfs_eemd(signal):
    try:
        signal = np.array(signal)
        imfs = EEMD().eemd(signal, max_imf=3)
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        return imfs[1:3]
    except Exception as e:
        print(e)
        return []


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



def marginal_frequency(signal):
    return np.sum(np.array(signal))


def mean_instAmp(signal):
    return (np.sum(signal)/len(signal))


# def minkowski_distance():


# Extract statistics values from imfs
def get_statistics_values(imfs):
    feat = []
    # For each imf compute 9 values and return it in a single vector. (5 values in this example)
    # Mean, maximum, minimum, standard deviation, variance, kurtosis, skewness, sum and median
    for ii, imf in enumerate(imfs):
        feat += [
            # stats.mean(imf),
            # np.var(imf),
            np.std(imf),
            kurtosis(imf),
            skew(imf),
            np.max(imf),
            # np.min(imf),
            # stats.median(imf)
        ]
    return feat


# Extract energy values from imfs
def get_energy_values(imfs):
    feat = []
    # For each imf compute
    # for imf in imfs:
    for ii, imf in enumerate(imfs):
        feat += [
            instantaneous_energy(imf),
            teager_energy(imf),
            # hfd(imf),
            # pfd(imf)
        ]
    return feat


def get_values_f(_vector):
    feat = []
    for ii, _vec in enumerate(_vector):
        feat += [
            np.std(_vec),
            max(_vec),
            skew(_vec),
            kurtosis(_vec),
            instantaneous_energy(_vec),
            teager_energy(_vec),
        ]
    return feat


# Extract features for each channel
# Extract feature from each instance
# def get_features(channels):
def get_features(instance):
    features_vector = []
    # channels = channels[0:2]
    for i, channel in enumerate(instance):
        if i < 5:
            # Compute the imf for ech channel
            imfs = get_imfs_emd(channel)
            # imfs = get_imfs_eemd(channel)

            # Compute feature values for imfs corresponding to one channel and join
            # features_vector += get_statistics_values(imfs)
            # features_vector += get_energy_values(imfs)
            features_vector += get_values_f(imfs)
    return features_vector


def get_Features(ch_imfs, sr):
    feature_vector = []

    for channel, bands in enumerate(ch_imfs):  # bands:(5, 3, 260)
        bands_instFreq, bands_instAmp = hilbert_transform(bands, sr)  # bands_instFreq:  (5, 3, 259)

        feat_vec = []
        bands_marginal_freq = []
        bands_ampMean = []

        for band, modes in enumerate(bands_instFreq):

            marginal_freq = []
            ampMean = []

            for i in range(len(modes)):
                feat_vec += [
                    marginal_frequency(modes[i]),
                    mean_instAmp(bands_instAmp[band][i])
                ]

            # bands_marginal_freq += marginal_freq
            # bands_ampMean += ampMean

        #feat_vec += [bands_marginal_freq, bands_ampMean]
        # feat_vec.append(bands_marginal_freq)  # ch_margin_freq:  (1, 15)
        # feat_vec.append(bands_ampMean)

        #feature_vector.append(feat_vec)
        feature_vector += feat_vec
    return feature_vector


def f_values_prodo(_vector, fs):
    feat = []
    band_instFreq, band_instAmp = hilbert_transform(_vector, fs)
    for i in range(len(band_instFreq)):
        feat += [
            marginal_frequency(band_instFreq[i]),
            mean_instAmp(band_instAmp[i])
        ]
    return feat


# ch_freq_bands: (2, 5, 260)
# feature_vector:  (2, 2, 15)
def get_Features_prodo(ch_freq_bands, fs):
    feature_vector = []
    # (2, 5, 3, 260)
    ch_imfs = get_imfs(ch_freq_bands)
    for channel, bands in enumerate(ch_imfs):
        for band, modes in enumerate(bands):
            feature_vector += f_values_prodo(modes, fs)
    return feature_vector

