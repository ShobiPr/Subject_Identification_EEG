from __future__ import division
import numpy as np
import pickle
import math
from PyEMD.EEMD import EEMD
from PyEMD.visualisation import Visualisation
import matplotlib.pyplot as plt
from featuresExtraction import get_features
from preprosessing import preprocessing
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


def get_dataset(n_subjects=1, n_sessions=1):
    sr = 200
    cutoff_lowpass = 50.0
    for subject in range(1, n_subjects + 1):  # 27
        for session in range(1, n_sessions + 1):  # 6
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            instance = preprocessing(cutoff_lowpass, 1, instances, sr)
    return instance  # 2 (data, target), data:9, target: 9


dataset = get_dataset()
t = np.linspace(0, 1.3, 260)

channel1 = dataset[0]

components = EEMD().eemd(channel1, max_imf=3)
imfs, res = components[:-1], components[-1]

vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=False)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()






