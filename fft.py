# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft
from scipy import signal
import warnings
from features import get_features
import warnings

warnings.filterwarnings("ignore")


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    _header = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _header = _rows
            _header = _header.split(',')
    return np.array(channels), np.array(_header[1:-1])


def notch_filter(instance, sr, f0=60.0, Q=10.0):
    filtered_instance = []
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, sr)
    for i, channel in enumerate(instance):
        filtered_instance.append(signal.filtfilt(b, a, channel))
    return np.array(filtered_instance)


def common_average_reference(instance):
    CAR = []
    for i, channel in enumerate(instance):
        CAR.append(channel - stats.mean(channel))
    return np.array(CAR)


def preprocessing(f_instance, instances, sr):
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = notch_filter(instance, sr)
    averaged_instance = common_average_reference(filtered_instance)
    """
    timep = 1.3  # Number of seconds
    nsamp = 260  # Number of samples
    t = np.linspace(0.0, timep, nsamp)
    fig, axes = plt.subplots(nrows=2)
    # plot time signal:
    axes[0].set_title("Signal")
    axes[0].plot(t, filtered_instance[4], color='C0')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    # plot different spectrum types:
    axes[1].set_title("Magnitude Spectrum")
    axes[1].magnitude_spectrum(filtered_instance[4], Fs=sr, color='C1')
    fig.tight_layout()
    plt.show()
    """

    """
    t = np.linspace(0.0, 1.3, 260)
    s = instance[0]
    s1 = filtered_instance[0]
    fft = np.fft.fft(s)
    fft1 = np.fft.fft(s1)
    T = t[1] - t[0]
    N = s.size
    f = np.linspace(0, 1 / T, N)

    fig, axes = plt.subplots(nrows=2)
    axes[0].set_title("instance")
    axes[0].plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    axes[1].set_title("filtered_instance")
    axes[1].plot(f[:N // 2], np.abs(fft1)[:N // 2] * 1 / N)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Amplitude")
    fig.tight_layout()
    plt.show()

    """
    t = np.linspace(0.0, 1.3, 260)
    s = averaged_instance[0]
    fft = np.fft.fft(s)
    T = t[1] - t[0]
    N = s.size

    f = np.linspace(0, 1/T, N)

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N)  # 1 / N is a normalization factor
    plt.show()

    return averaged_instance


def get_dataset(n_subjects=1, n_sessions=1):
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, n_subjects + 1):  # 27
        for session in range(1, n_sessions + 1):  # 6
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):  # len(instances) 60 instances
                instance = preprocessing(f_instance, instances, sr)
                ch_fs_instances.append(get_features(instance))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}  # 2 (data, target), data:9, target: 9


dataset = get_dataset(n_subjects=3, n_sessions=1)

for i, ii in enumerate(dataset['data']):
    color = "red" if dataset['target'][i] == "subject_1" else (
        "green" if dataset['target'][i] == "subject_2" else "blue")
    plt.plot(ii, color)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.show()
