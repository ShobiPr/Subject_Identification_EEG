# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
from filters import butter_bandpass_filter, notch_filter
import scipy.io as spio
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def preprocessing_P300(instances, f_instance, fs, lowcut=0.5, highcut=70.0, order=6):
    """ Notch filter, f0 = 50, Q=10.0
        bandpass : [0.5-70.0] Hz order:6 """
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = []
    for i, channel in enumerate(instance):
        notch_f = notch_filter(channel, fs, f0=50.0, Q=20.0)
        filtered_instance.append(butter_bandpass_filter(notch_f, fs, lowcut, highcut, order))
    return np.array(filtered_instance)


def preprocessing_P300(instances, f_instance, fs, ch, lowcut=0.5, highcut=70.0, order=6):
    """ Notch filter, f0 = 50, Q=10.0
        bandpass : [0.5-70.0] Hz order:6 """
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    ins = instance[ch, :]
    filtered_instance = []
    for i, channel in enumerate(ins):
        notch_f = notch_filter(channel, fs, f0=50.0, Q=20.0)
        filtered_instance.append(butter_bandpass_filter(notch_f, fs, lowcut, highcut, order))
    return np.array(filtered_instance)


def preprocessing_resting(sub_instance):
    """ Offset: 4200  """
    filtered_instance = []
    for i, channel in enumerate(sub_instance):
        signal = channel - 4200
        filtered_instance.append(signal)
    return np.array(filtered_instance)