# -*- coding: utf-8 -*-
import scipy.io as spio
import numpy as np
from features import get_features_emd
from preprocessing import preprocessing_resting


import logging
import warnings


warnings.filterwarnings("ignore")
logging.basicConfig(filename='minkowski_resting_State.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

ch_fs_instances = []
ch_tags_instances = []

samples_subject = 21000 # samples per subject
samples_trial = 7000  # samples per trial
no_subjects = 8
no_ch = 14
sr = 128


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


instance_len = sr * 2  # to create sub instances of 2 sec
EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
_labels = EEG_data[14, :]  # len(168000)

for subject in range(1, no_subjects + 1):
    s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999],

    for s_instance in range(s_index[0], s_index[1] + 1, samples_subject):
        _instance = EEG_data[:no_ch, s_instance:s_instance + samples_trial]  # (14, 7000)
        max_instances = _instance.shape[1] / instance_len  # this is not necessary, but I added it just FYI, 27

        for _i in range(0, 20):  # sub instances
            if _i < max_instances:
                index_start, index_end = instance_len * _i, instance_len * (_i + 1)
                sub_instance = _instance[:, index_start:index_end]
                sub_ins = preprocessing_resting(sub_instance)
                ch_fs_instances.append(get_features_emd(sub_ins, sr))



