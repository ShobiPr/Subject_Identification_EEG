# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy.io as spio
from features import get_features_emd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def get_dataset():
    # samples_subject = 21000 # samples per subject
    samples_trial = 7000  # samples per trial
    no_subjects = 8
    no_ch = 14
    sr = 128
    ch_fs_instances = []
    ch_tags_instances = []
    sub_ins_m = []
    instance_len = sr * 2  # to create sub instances of 2 sec
    EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
    _labels = EEG_data[14, :]  # len(168000)
    for subject in range(1, no_subjects + 1):
        s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999]
        _instance = EEG_data[:no_ch, s_index[0]:s_index[0] + samples_trial]  # (14, 7000)
        max_instances = _instance.shape[1] / instance_len  # this is not necessary, but I added it just FYI, 27
        for _i in range(0, 20): #range(0, 20):  # sub instances
            if _i < max_instances:
                index_start, index_end = instance_len * _i, instance_len * (_i + 1)
                sub_instance = _instance[:, index_start:index_end]
                print('sub-instance {0} with shape {1} for subject {2}'.format(_i, sub_instance.shape, subject))
            ch_fs_instances.append(get_features_emd(sub_instance, sr))
    return {"data": ch_fs_instances}


data = get_dataset()



for i, ii in enumerate(data['data']):
    plt.plot(ii, linewidth=0.8)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.title('Statistical features')
plt.savefig('features_stat.svg', format='svg')
plt.show()