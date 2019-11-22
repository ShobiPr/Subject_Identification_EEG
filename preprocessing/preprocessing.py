# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
from scipy import signal
import matplotlib.pyplot as plt
from preprocessing.filters import butter_bandpass_filter
import math
import scipy.io as spio
import warnings

warnings.filterwarnings("ignore")


def notch_filter(instance, sr, f0=60.0, Q=10.0):
    filtered_instance = []
    # Design notch filter
    b, a = signal.iirnotch(f0, Q, sr)
    for i, channel in enumerate(instance):
        filtered_instance.append(signal.filtfilt(b, a, channel))
    return np.array(filtered_instance)


def notch_filter(s, sr, f0=50.0, Q=10.0):
    b, a = signal.iirnotch(f0, Q, sr)
    return signal.filtfilt(b, a, s)


def common_average_reference(instance):
    CAR = []
    for i, channel in enumerate(instance):
        CAR.append(channel - stats.mean(channel))
    return np.array(CAR)


def preprocessing_P300(instances, f_instance, fs, lowcut=0.05, highcut=70.0, order=4):
    """ Notch filter, f0 = 50, Q=10.0
        bandpass : [0.5-70.0] """
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = []
    for i, channel in enumerate(instance):
        notch_f = notch_filter(channel, fs, f0=50-0, Q=10.0)
        filtered_instance.append(butter_bandpass_filter(notch_f, lowcut, highcut, fs, order))
    return np.array(filtered_instance)


def preprocessing_resting(sub_instance, sr, lowcut, highcut, order=4):
    """ Offset: 4200  """
    filtered_instance = []
    for i, channel in enumerate(sub_instance):
        signal = channel - 4200
        filtered_instance.append(signal)
    return np.array(filtered_instance)


"""
def get_samples(_index, s_s_chs, sr, _size=1.3):
    return s_s_chs[_index:int(math.ceil(_index + (_size * sr)))][:]


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


def get_dataset():
    sr = 200
    s_s_chs, _header = get_subdataset()
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)  # (60, 260, 59)
    instance = np.array(instances[1, :, 1:-1]).transpose()  # (57, 260)
    return instance

# --------------------------------------------------------------------------

def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def get_signals_resiting():
    # samples_subject = 21000 # samples per subject
    samples_trial = 7000  # samples per trial
    no_subjects = 8
    no_ch = 14
    sr = 128
    instance_len = sr * 2  # to create sub instances of 2 sec
    EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T
    _labels = EEG_data[14, :]
    data = []
    for subject in range(1, no_subjects + 1):
        s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]
        _instance = EEG_data[0, s_index[0]:s_index[0] + samples_trial]
        index_start, index_end = instance_len * 0, instance_len * (0 + 1)
        sub_instance = _instance[index_start:index_end]
        data.append(sub_instance)
        print('sub-instance with shape {0} for subject {1}'.format(sub_instance.shape, subject))
            #ch_fs_instances.append(get_features(sub_instance))
            # ch_tags_instances.append('subject_{0}'.format(subject))
    return data


def get_signals_P300():
    sr = 200
    lowcut = 0.1
    highcut = 70.0
    order = 4
    session = 1
    data = []
    for subject in range(1, 27):  # 26
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        all_channels = get_samples(_index[0], s_s_chs, sr)
        data.append(butter_bandpass_filter(all_channels[:, 55], lowcut, highcut, sr, order=order))
        print("subject: ", subject)
    return data


for i in range(0, len(data)):
    plt.plot(data[i], label='Subject {0}'.format(i + 1), linewidth=0.9)
plt.title('Filtered EEG-signals [P300]')
plt.xlabel('Samples')
plt.ylabel('Amplitude [mV]')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# plt.savefig('.svg', format='svg')
plt.show()
"""
