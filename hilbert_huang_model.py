from esemble_emd import get_imfs
from filters import get_frequency_bands
import matplotlib.pyplot as plt
from hilbert_transform import hilbert_transform
from features import get_Features
from PyEMD import EEMD
import math
from classefiers import SVM
import numpy as np


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
    ch_features_instances = []
    ch_tags_instances = []
    n_channel = 2
    for subject in range(1, n_subjects + 1):
        for session in range(1, n_sessions + 1):
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):
                instance = np.array(instances[f_instance, :, 1:-1]).transpose()
                ch_freq_bands = get_frequency_bands(instance, n_channel, sr)
                ch_imfs = get_imfs(ch_freq_bands)  # (2, 5, 3, 260)
                ch_features_instances.append(get_Features(ch_imfs, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_features_instances, "target": ch_tags_instances}

n_subjects = 3
n_sessions = 2
dataset = get_dataset(n_subjects, n_sessions)


C_F_V = 3
result = SVM(dataset['data'], dataset['target'], C_F_V)
print("{0}, accuracy {1}".format(result['classifier'], result['accuracy']))


for i, ii in enumerate(dataset['data']):
    color = "red" if dataset['target'][i] == "subject_1" else (
        "green" if dataset['target'][i] == "subject_2" else "blue")
    plt.plot(ii, color)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# ch_freq_bands: (2, 5, 260)
# [ch_imfs]: (2, 5, 3, 260)
# feature_vector:  (2, 2, 15)