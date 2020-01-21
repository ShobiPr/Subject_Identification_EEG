# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from dataset_P300 import get_subdataset, get_samples
from features import get_features_sub_bands

import pickle
import logging

logging.basicConfig(filename='Validation_subband_ch7_energy_hht_nopre.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')


def get_dataset(subject, ins):
    sr = 200
    ch = [1, 42, 46, 51, 52, 54, 55]
    # ch =  ch = [0, 1, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 36, 36, 38, 40,
    # 41, 42, 44, 46, 48, 50, 51, 52, 53, 54, 55]  32 channels
    session = 5
    ch_fs_instances = []
    ch_tags_instances = []
    
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, ins):
        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
        ins7 = instance[ch, :]
        ch_fs_instances.append(get_features_sub_bands(ins7, sr))  # CHANNELS: 14
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


for ins in [10, 20, 40, 60]:
    logging.info("----------- energy + hht ins: {0}---------".format(ins))
    for subject in range(1, 27):
        dataset = get_dataset(subject, ins)
        file_name = 'P300_subband_ch7_energy_nopre_hht_ins%2d.sav' % ins
        model = open(file_name, 'rb')
        clf = pickle.load(model)
        eval_model(dataset, clf)
    logging.info("\n")

