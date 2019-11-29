# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
from filters import butter_bandpass_filter, notch_filter
import scipy.io as spio
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def preprocessing_P300(instances, f_instance, fs, ch, lowcut=0.5, highcut=70.0, order=6):
    """ Notch filter, f0 = 50, Q=10.0
        bandpass : [0.5-70.0] Hz order:6 """
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    ins = instance[ch, :]
    filtered_instance = []
    for i, channel in enumerate(ins):
        notch_f = notch_filter(channel, fs, f0=50.0, Q=20.0)
        filtered_instance.append(butter_bandpass_filter(notch_f, fs, lowcut, highcut, order))
    return np.array(filtered_instance)


def preprocessing_resting(sub_instance):
    """ Offset: 4200  """
    filtered_instance = []
    for i, channel in enumerate(sub_instance):
        signal = channel - 4200
        filtered_instance.append(signal)
    return np.array(filtered_instance)


# --------------------------------------------------------------------

"""
def get_samples(_index, s_s_chs, sr, _size=1.3):
    return s_s_chs[_index:int(math.ceil(_index + (_size * sr)))][:]


def get_subdataset(_S, Sess):
    _file = 'P300/train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _headers = _rows
        if i > 90000:
            return np.array(channels)
    return np.array(channels)


def get_signals_P300():
    sr = 200
    session = 1
    r_data = []
    n_data = []
    b_data = []
    p_data = []
    for subject in range(1, 27):  # 26
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        instances = get_samples(_index[0], s_s_chs, sr)  # first instance
        for f_instance in range(0, 20):
            preprocessing_P300(instances, f_instance, sr)
            ins8 = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
            ch_fs_instances.append(get_features_emd(ins8, sr))

        m = np.mean(ch_fs_instances, axis=0)
        sub_ins_m.append(m)

        #raw data
        r_data.append(instance[:, 55])

        # Notch
        notch_f = notch_filter(instance[:, 55], sr, f0=50.0, Q=20.0)
        n_data.append(notch_f)

        # Bandpasss
        bandpass = butter_bandpass_filter(instance[:, 55], sr)
        b_data.append(bandpass)

        # Preprocessing
        p_data.append(butter_bandpass_filter(notch_f, sr))
        print("subject: ", subject)
    return {'raw': r_data, 'notch': n_data, 'bandpass': b_data, 'preprocessed': p_data}


# --------------------------------------------------------------------------

def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def get_signals_resting():
    # samples_subject = 21000 # samples per subject
    samples_trial = 7000  # samples per trial
    no_subjects = 8
    no_ch = 14
    sr = 128
    instance_len = sr * 2  # to create sub instances of 2 sec
    EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T
    _labels = EEG_data[14, :]
    data = []
    preprocessed = []
    for subject in range(1, no_subjects + 1):
        s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]
        _instance = EEG_data[0, s_index[0]:s_index[0] + samples_trial]
        index_start, index_end = instance_len * 0, instance_len * (0 + 1)
        sub_instance = _instance[index_start:index_end]
        data.append(sub_instance)
        preprocessed.append(preprocessing_resting(sub_instance))
        print('sub-instance with shape {0} for subject {1}'.format(sub_instance.shape, subject))
    return {'data': data, 'preprocessed': preprocessed}

# --------------------------------------------------------------------------

def plot_P300():
    dataset = get_signals_P300()

    raw = dataset['raw']
    plt.subplot(4, 1, 1)
    for i in range(0, len(raw)):
        plt.plot(raw[i], linewidth=0.9)
    plt.title('Raw EEG-signals [P300]')
    plt.ylabel('Amplitude [mV]')
    plt.grid()


    plt.subplot(4, 1, 2)
    notch = dataset['notch']
    for i in range(0, len(notch)):
        plt.plot(notch[i], linewidth=0.9)
    plt.title('Notch filtered EEG-signals [P300]')
    plt.ylabel('Amplitude [mV]')
    plt.grid()


    plt.subplot(4, 1, 3)
    bandpass = dataset['bandpass']
    for i in range(0, len(bandpass)):
        plt.plot(bandpass[i], linewidth=0.9)
    plt.title('Bandpass filtered EEG-signals [P300]')
    plt.ylabel('Amplitude [mV]')
    plt.grid()


    plt.subplot(4, 1, 4)
    preprocessed = dataset['preprocessed']
    for i in range(0, len(preprocessed)):
        plt.plot(preprocessed[i], linewidth=0.9)
    plt.title('Preprocessed EEG-signals [P300]')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude [mV]')
    plt.grid()

    plt.show()


def plot_resting():
    dataset = get_signals_resting()

    data = dataset['data']
    plt.subplot(2, 1, 1)
    for i in range(0, len(data)):
        plt.plot(data[i], linewidth=0.9)
    plt.title('Raw EEG-signals [Resting state]')
    plt.ylabel('Amplitude [mV]')
    plt.grid()


    plt.subplot(2, 1, 2)
    pre = dataset['preprocessed']
    for i in range(0, len(pre)):
        plt.plot(pre[i], linewidth=0.9)
    plt.title('Preprocessed EEG-signals [Resting state]')
    plt.ylabel('Amplitude [mV]')
    plt.grid()

    plt.show()

"""