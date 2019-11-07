# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pylab as plt
import math
from preprosessing import preprocessing
import warnings

from pyhht import EMD
from pyhht.visualization import plot_imfs

warnings.filterwarnings("ignore")


def get_subdataset(_S=1, Sess=1):
    _file = 'train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _headers = _rows
    return np.array(channels)


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_features(instance):
    features_vector = []
    for i, channel in enumerate(instance):
        if i < 1:
            get_imfs_emd(channel)


def get_imfs_emd(signal):
    # decomposer_signal = EMD(signal, fixe=100, n_imfs=2)
    decomposer_signal = EMD(signal)
    imfs = decomposer_signal.decompose()

    nIMFs = len(imfs)

    print("nIMFS: ", nIMFs)
    for n in range(nIMFs):
        plt.subplot(nIMFs + 4, 1, n + 2)
        plt.plot(imfs[n])
        plt.ylabel("IMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)
    plt.show()


def get_dataset():
    sr = 200
    lowcut = 0.01
    highcut = 50.0
    order = 4
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 2):  # 3
        for session in range(1, 2):  # 1
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):  # 10 instances
                instance = preprocessing(lowcut, highcut, f_instance, order, instances, sr)
                get_features(instance)


get_dataset()
