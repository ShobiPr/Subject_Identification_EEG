import numpy as np
import math


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'test/Data_S%02d_Sess%02d.csv' % (_S, Sess)
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


channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7',
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2','FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'T8', 'TP7', 'CP5', 'CP3', 'CP1','CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
            'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2', 'EOG']

ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7',
      'CP5', 'CP1', 'CP2', 'CP6', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']


position = [0, 1, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 36, 36, 38, 40, 41, 42, 44, 46, 48, 50, 51, 52, 53, 54, 55]

pos = []
for i in range(0, len(channels)):
    for j in range(0, len(position)):
        if channels[i] == ch[j]:
            pos.append(channels[i])