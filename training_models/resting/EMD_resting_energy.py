import numpy as np
import scipy.io as spio

from features import get_imfs_emd, get_energy_values
from preprocessing.preprocessing import preprocessing_resting
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_resting_energy.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def get_features_emd(instance):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_emd(channel)
        if len(imfs) > 1:
            features_vector += get_energy_values(imfs)
    return features_vector


def main():
    logging.info(" ***** CHANNELS:8, FEATURES: energy ***** \n")
    logging.info(" ---------- With preprocessing ---------- \n")
    INSTANCES = [10, 20, 40, 60]
    sr = 200
    lowcut = 0.1
    highcut = 70.0
    order = 4
    for ins in INSTANCES:
        logging.info(" -------- Instance: {0} --------".format(ins))
        ch_fs_instances = []
        ch_tags_instances = []
        samples_trial = 7000  # samples per trial
        no_subjects = 8
        no_ch = 14
        sr = 128
        lowcut = 0.5
        highcut = 60.0
        order = 4
        instance_len = sr * 2  # to create sub instances of 2 sec
        EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
        _labels = EEG_data[14, :]  # len(168000)
        for subject in range(1, no_subjects + 1):
            s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999]
            for s_instance in range(s_index[0], s_index[1] + 1, samples_trial):
                _instance = EEG_data[:no_ch, s_instance:s_instance + samples_trial]  # (14, 7000)
                max_instances = _instance.shape[1] / instance_len  # 27
                for _i in range(0, 20):  # sub instances
                    if _i < max_instances:
                        index_start, index_end = instance_len * _i, instance_len * (_i + 1)  # start = 0, end = 256
                        sub_instance = _instance[:, index_start:index_end]
                        ins = preprocessing_resting(sub_instance, sr, lowcut, highcut, order=order)
                        print("preprocessed")
                        # print('sub-instance {0} with shape {1} for subject {2}'.format(_i, sub_instance.shape, subject))
                        ch_fs_instances.append(get_features_emd(sub_instance))
                        ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'EMD_resting_energy_ins%02d.sav' % (ins)
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()