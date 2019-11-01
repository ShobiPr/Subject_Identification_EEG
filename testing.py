# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft
from scipy import signal
import warnings
from preprosessing import preprocessing
from featuresExtraction import get_features
import warnings
from scipy.signal import hilbert
from pyhht import EMD
from pyhht.utils import inst_freq

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

def instantaneous_frequency(signal):
    """Calculate instantaneous frequency from HHT"""
    hs = hilbert(signal)  # Hilbert Transform
    instant_freq, timestamps = inst_freq(hs)
    return instant_freq


def instantaneous_amplitude(signal):
    """Calculate instantaneous ampltiude from HHT"""
    hs = hilbert(signal)  # Hilbert Transform
    ampl = np.abs(hs)  # Calculate amplitude
    return ampl


sr = 200
cutoff_lowpass = 50.0
ch_fs_instances = []
ch_tags_instances = []
s_s_chs, _header = get_subdataset(2,1)
_index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
instances = get_samples(_index, s_s_chs, sr)
instance = preprocessing(cutoff_lowpass, 1, instances, sr)  # (57, 260)
ch_fs_instances.append(get_features(instance))



plt.subplot(4, 1, 1)
amp, t = instantaneous_amplitude(instance[0])
plt.plot(t, amp)

plt.subplot(4, 1, 2)
amp, t = instantaneous_amplitude(instance[1])
plt.plot(t, amp)

plt.subplot(4, 1, 3)
amp, t = instantaneous_amplitude(instance[2])
plt.plot(t, amp)

plt.subplot(4, 1, 4)
amp, t = instantaneous_amplitude(instance[3])
plt.plot(t, amp)
plt.show()

