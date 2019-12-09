# -*- coding: utf-8 -*-
import scipy.io as spio
import numpy as np
from features import get_features_emd
from classefiers import selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_resting_stat_nopre.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def main():
    logging.info(" ***** Resting state, EMD , FEATURES: stat ***** ")
    logging.info(" ---------- No preprocessing ---------- \n \n")

    ch_fs_instances = []
    ch_tags_instances = []

    samples_subject = 21000 # samples per subject
    samples_trial = 7000  # samples per trial
    no_subjects = 8
    no_ch = 14
    sr = 128

    for ins in [10, 20]:
        logging.info(" -------- Instance: {0} --------".format(ins))
        instance_len = sr * 2  # to create sub instances of 2 sec
        EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
        _labels = EEG_data[14, :]  # len(168000)

        for subject in range(1, no_subjects + 1):
            s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999],

            for s_instance in range(s_index[0], s_index[0] + samples_trial + 1, samples_trial):
                _instance = EEG_data[:no_ch, s_instance:s_instance + samples_trial]  # (14, 7000)
                max_instances = _instance.shape[1] / instance_len  # this is not necessary, but I added it just FYI, 27

                for _i in range(0, ins):  # sub instances
                    if _i < max_instances:
                        index_start, index_end = instance_len * _i, instance_len * (_i + 1)
                        sub_instance = _instance[:, index_start:index_end]
                        ch_fs_instances.append(get_features_emd(sub_instance))
                        ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'EMD_resting_stat_nopre_ins%02d.sav' % ins
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()
