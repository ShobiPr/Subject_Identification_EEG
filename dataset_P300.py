from preprocessing import preprocessing_P300
import numpy as np
import math


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'P300/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _headers = _rows
    return np.array(channels)


def get_dataset(n_subjects, n_sessions, n_ins):
    sr = 200
    s_instance = []
    for subject in range(1, n_subjects + 1):  # 26
        for session in range(1, n_sessions + 1):  # 4
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(0, n_ins):
                instance = preprocessing_P300(instances, f_instance, sr)
                ins1 = instance[55]
                s_instance.append(ins1)
    return s_instance
