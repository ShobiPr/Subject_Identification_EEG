import pandas as pd
import numpy as np
import pywt
from scipy.stats import kurtosis
from pyhht import EMD
import matplotlib.pyplot as plt
from classefiers import selector
import logging

logging.basicConfig(filename='EMD_resting_test.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')


def get_imfs_emd(signal):
    try:
        # decomposer_signal = EMD(signal, fixe=100, n_imfs=2)
        decomposer_signal = EMD(signal, n_imfs=4)
        imfs = decomposer_signal.decompose()
        if len(imfs) < 2:
            print("imfs {} +++++++++++++++++++++++++++++++++++++++".format(len(imfs)))
            raise ValueError("imfs {}".format(len(imfs)))
        return imfs[:2]
    except Exception as e:
        raise e


def get_statistics_values(imfs):
    feat = []
    for ii, imf in enumerate(imfs):
        feat += [
            np.var(imf),
            np.std(imf),
            kurtosis(imf),
            np.max(imf),
            np.min(imf),
        ]
    return feat

def get_features(instance, fs):
    features_vector = []
    for ch, channel in instance.iterrows():
        signal = channel - 4200
        imfs = get_imfs_emd(channel)
        features_vector += get_statistics_values(imfs)
    return features_vector


def read_csv(_S=1, Sess=1):
    _file = 'resting/Resting_S%d_Sess%d.csv' % (_S, Sess)
    data = pd.read_csv(_file, header=None, index_col=False)
    return data.transpose()


def waveletTransform(signal):
    labels = ['cA5', 'cd5', 'cd4', 'cd3', 'cd2', 'cd1']
    coefficients = pywt.wavedec(signal, 'sym7', level=5)
    fig, axs = plt.subplots(nrows=len(coefficients))
    print(len(coefficients))
    for i in range(len(coefficients)):
        # explicitly create and save the secondary axis
        axs[i].plot(coefficients[i])
        axs[i].set_ylabel(labels[i])

    plt.show()

data = read_csv()
signal = data[0] - 4200
waveletTransform(signal)


"""
fs = 128
ch_fs_instances = []
ch_tags_instances = []
for subject in range(1, 9):
    for session in range (1, 3):
        s_s_chs = read_csv(subject, session)
        ch_fs_instances.append(get_features(s_s_chs, fs))
        ch_tags_instances.append('subject_{0}'.format(subject))
dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

dataTraining = dataset['data']
targetTraining = dataset['target']
result = selector(dataTraining, targetTraining)

logging.info("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))
"""
