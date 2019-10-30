from __future__ import division
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from classefiers import selector
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
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, n_subjects + 1):  # 27
        for session in range(1, n_sessions + 1):  # 6
            s_s_chs, _header = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(1, 2):  # len(instances) 60 instances
                instance = preprocessing(f_instance, instances, sr)
                ch_fs_instances.append(get_features(instance))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}  # 2 (data, target), data:9, target: 9


dataset = get_dataset(n_subjects=3, n_sessions=2)


if __name__ == '__main__':
    n_subjects = 3
    n_sessions = 5
    dataset = get_dataset(n_subjects=n_subjects, n_sessions=n_sessions)

    C_F_V = 3
    RANDOM_STATE = 0
    dataTraining = dataset['data']
    targetTraining = dataset['target']
    result = selector(dataTraining, targetTraining)

    print("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))

    # saving the model
    # model_name = 'clf.sav'
    # pickle.dump(result["clf"], open(model_name, 'wb'))
