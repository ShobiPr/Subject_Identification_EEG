# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pickle
from dataset_P300 import get_subdataset, get_samples
from features import get_features_sub_bands
from preprocessing import preprocessing_P300

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='Validation_sub_bands_P300_ch56_HHT.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

def get_dataset(subject, ins):
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    session = 5
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, ins):
        instance = preprocessing_P300(instances, f_instance, sr)
        ch_fs_instances.append(get_features_sub_bands(instance, sr))  # CHANNELS: 8
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
    logging.info("----------- HHT ins: {0}---------".format(ins))
    for subject in range(1, 27):
        dataset = get_dataset(subject, ins)
        file_name = 'sub_bands_P300_ch56_HHT_ins%2d.sav' % ins
        model = open(file_name, 'rb')
        clf = pickle.load(model)
        eval_model(dataset, clf)
    logging.info("\n")
