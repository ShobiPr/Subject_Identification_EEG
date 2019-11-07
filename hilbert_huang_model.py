from filters import get_frequency_bands, plot_freq_bands
from features import get_features_eemd
import math
import numpy as np


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


def get_dataset_HHT():
    sr = 200
    ch_start = 10
    ch_stop = 12
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 3):  # 3
        for session in range(1, 2):  # 1
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):  # len(instances) 60 instances
                fb = get_frequency_bands(instances, f_instance, ch_start, ch_stop, sr)
                ch_fs_instances.append(get_features_eemd(fb, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
                #return ch_fs_instances
                return {"data": ch_fs_instances, "target": ch_tags_instances}


a = get_dataset_HHT()
