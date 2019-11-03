from PyEMD import EEMD, Visualisation
import numpy as np
import math
import matplotlib.pyplot as plt
from preprosessing import preprocessing

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

"""
sr = 200
lowcut = 0.01
highcut = 50.0
order = 6
s_s_chs, _header = get_subdataset()
_index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
instances = get_samples(_index, s_s_chs, sr)
instance = preprocessing(lowcut, highcut, order, 1, instances, sr)

t = np.linspace(0, 1.3, 260)
s = instance[1]
components = EEMD()
eIMFs = components.eemd(s, max_imf=2)
IMF = eIMFs[1]
"""

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





