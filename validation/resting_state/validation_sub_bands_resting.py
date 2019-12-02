# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pickle
from features import get_features_sub_bands
from preprocessing import preprocessing_resting
import scipy.io as spio
import logging

logging.basicConfig(filename='Validatoin_sub_bands_HHT_ins20.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')


def consecutive_index(data, _value, stepsize=1):
    data = np.where(data == _value)[0]
    result = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return [[min(a), max(a)] for a in result]


def get_dataset(subject, ins):
    ch_fs_instances = []
    ch_tags_instances = []
    samples_trial = 7000  # samples per trial
    no_ch = 14
    sr = 128
    instance_len = sr * 2  # to create sub instances of 2 sec
    EEG_data = spio.loadmat('EID-M.mat', squeeze_me=True)['eeg_close_ubicomp_8sub'].T  # (15, 168000)
    _labels = EEG_data[14, :]  # len(168000)
    s_index = consecutive_index(EEG_data[no_ch, :], subject)[0]  # [0, 20999],

    for s_instance in range(s_index[0] + 7000, s_index[1] + 1, samples_trial):
        _instance = EEG_data[:no_ch, s_instance:s_instance + samples_trial]  # (14, 7000)
        max_instances = _instance.shape[1] / instance_len  # this is not necessary, but I added it just FYI, 27

        for _i in range(0, ins):  # sub instances
            if _i < max_instances:
                index_start, index_end = instance_len * _i, instance_len * (_i + 1)
                sub_instance = _instance[:, index_start:index_end]
                sub_ins = preprocessing_resting(sub_instance)
                ch_fs_instances.append(get_features_sub_bands(sub_ins, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}


def eval_model(dataset, clf):
    false_accepted = 0
    Ok_accepted = 0
    total_tags = len(dataset['target'])
    for i, unk_entry in enumerate(dataset['target']):
        true_tag = dataset['target'][i]
        feature_vector = np.array([dataset['data'][i]])
        prediction = clf.predict(feature_vector)[0]
        accuracy = max(max(clf.predict_proba(feature_vector)))
        result_ = "True label: {0},  prediction: {1}, accuracy: {2}".format(true_tag, prediction, accuracy)
        logging.info(result_)
        if true_tag == prediction:
            Ok_accepted += 1
        else:
            false_accepted += 1
    logging.info('Ok_accepted {0}'.format(Ok_accepted))
    logging.info('false_accepted {0}'.format(false_accepted))
    logging.info('accuracy of Ok_accepted {0}'.format(round(Ok_accepted / total_tags, 10)))
    logging.info('accuracy of false_accepted {0}'.format(round(false_accepted / total_tags, 10)))
    logging.info("\n")


no_subjects = 8
for ins in [10, 20]:
    for subject in range(1, no_subjects + 1):
        dataset = get_dataset(subject, ins)
        file_name = 'sub_bands_resting_HHT_ins%2d.sav' % ins
        model = open(file_name, 'rb')
        clf = pickle.load(model)
        eval_model(dataset, clf)


