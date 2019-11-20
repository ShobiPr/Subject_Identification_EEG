# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
from scipy import signal
from filters import butter_bandpass_filter

import warnings
warnings.filterwarnings("ignore")

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


def preprocessing(lowcut, highcut, f_instance, instances, order, fs):
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = []
    for i, channel in enumerate(instance):
        filtered_instance.append(butter_bandpass_filter(channel, lowcut, highcut, fs, order=order))
    return np.array(filtered_instance)
