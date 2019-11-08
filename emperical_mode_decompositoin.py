# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pylab as plt
import math
from preprosessing import preprocessing
from features import get_features_emd
import warnings

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


def get_dataset_EMD():
    sr = 200
    lowcut = 0.01
    highcut = 50.0
    order = 4
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 6):  # 3
        for session in range(1, 2):  # 1
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 11):  # 10 instances
                # instance = np.array(instances[f_instance, :, 1:-1]).transpose()
                instance = preprocessing(lowcut, highcut, f_instance, order, instances, sr)
                ch_fs_instances.append(get_features_emd(instance,sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}


