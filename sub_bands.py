import numpy as np
import scipy.io as spio
from classefiers import selector
from features import get_features_bands

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='resting_sub_bands_TEST.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def main():
    logging.info(" ***** CHANNELS:8, FEATURES: energy ***** \n")
    INSTANCES = [10, 20]
    sr = 200
    for ins in INSTANCES:
        logging.info(" -------- Instance: {0} --------".format(ins))
        ch_fs_instances = []
        ch_tags_instances = []
        samples_trial = 7000  # samples per trial
        no_subjects = 8
        no_ch = 14
        sr = 128
        instance_len = sr * 2  # to create sub instances of 2 sec
        EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
        _labels = EEG_data[14, :]  # len(168000)
        for subject in range(1, no_subjects + 1):
            s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999]
            _instance = EEG_data[:no_ch, s_index[0]:s_index[0] + samples_trial]  # (14, 7000)
            print("_instance: ", np.shape(_instance))
            for _i in range(0, ins):  # sub instances
                index_start, index_end = instance_len * _i, instance_len * (_i + 1)  # start = 0, end = 256
                sub_instance = _instance[:, index_start:index_end]  # (14, 256)
                ch_fs_instances.append(get_features_bands(sub_instance, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
            print("Subject: ", subject)
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        print("classification started")
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

if __name__ == '__main__':
    main()