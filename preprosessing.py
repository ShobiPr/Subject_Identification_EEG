# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter
from scipy import signal
from featuresExtraction import get_features
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


def butter_lowpass_filter(data, fs, cutoff=50.0, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def preprocessing(f_instance, instances, sr):
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = []
    for i, channel in enumerate(instance):
        filtered_instance.append(butter_lowpass_filter(channel, sr))
    return np.array(filtered_instance)


def get_dataset():
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 4):  # 27
        for session in range(1, 2):  # 6
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):  # len(instances) 60 instances
                instance = preprocessing(f_instance, instances, sr)
                ch_fs_instances.append(get_features(instance))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}


dataset = get_dataset()

for i, ii in enumerate(dataset['data']):
    color = "red" if dataset['target'][i] == "subject_1" else (
        "green" if dataset['target'][i] == "subject_2" else "blue")
    plt.plot(ii, color)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.show()
