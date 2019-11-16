from dataset import get_subdataset, get_samples
from filters import get_frequency_bands
from features import get_features_eemd
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_P300_ch57_stat.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')

if __name__ == "__main__":

    logging.info(" ***** CHANNELS:57, FEATURES: STAT ***** \n")
    INSTANCES = [10]
    for ins in INSTANCES:

        logging.info(" -------- Instance: {0} --------".format(ins))
        sr = 200
        ch_start = 2
        ch_stop = 3
        ch_fs_instances = []
        ch_tags_instances = []
        for subject in range(1, 27):  # 26
            for session in range(1, 5):  # 4
                s_s_chs = get_subdataset(subject, session)
                _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
                instances = get_samples(_index, s_s_chs, sr)
                for f_instance in range(0, ins):  # INSTANCES = [10, 20, 30, 40]
                    freq_bands = get_frequency_bands(instances, f_instance, ch_start, ch_stop, sr)
                    ch_fs_instances.append(get_features_eemd(freq_bands, sr))
                    ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))

        # saving the model
        # model_name = 'EMD_ch57_stat_ins%02d.sav' % (ins)
        # pickle.dump(result["model"], open(model_name, 'wb'))

# -----------------------------------------------------

"""

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
            for f_instance in range(1, 11):  # len(instances) 60 instances
                fb = get_frequency_bands(instances, f_instance, ch_start, ch_stop, sr)
                ch_fs_instances.append(get_features_eemd(fb, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}

-----------------------------------------

# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pylab as plt
import math
from filters import get_frequency_bands
from preprosessing import preprocessing
from features import get_features_eemd
import warnings

warnings.filterwarnings("ignore")


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


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_dataset_eemd():
    sr = 200
    ch_start = 2
    ch_stop = 3
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 6):  # 26
        for session in range(1, 2):  # 4
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(0, 3):  # 20 instances
                freq_bands = get_frequency_bands(instances, f_instance, ch_start, ch_stop, sr)
                ch_fs_instances.append(get_features_eemd(freq_bands, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}
"""
