# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pickle
import math
from preprosessing import preprocessing
from features import get_features_emd
import logging

logging.basicConfig(filename='validation_EMD_HHT_ch14_ins40.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'P300/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    _header = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _header = _rows
            _header = _header.split(',')
    return np.array(channels), np.array(_header[1:-1])


def get_dataset(subject, session=5):
    sr = 200
    cutoff = 50.0
    order = 6
    ch_fs_instances = []
    ch_tags_instances = []
    s_s_chs, _header = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, 40):  # len(instances) 60 instances
        instance = preprocessing(cutoff, f_instance, instances, order, sr)
        ins14 = instance[[4, 5, 7, 9, 13, 15, 17, 23, 25, 33, 43, 51, 55, 56], :]
        #ins8 = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
        ch_fs_instances.append(get_features_emd(ins14, sr))
        ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}


def eval_model(dataset, clf):
    false_accepted = 0
    Ok_accepted = 0
    total_tags = len(dataset['target'])
    for i, unk_entry in enumerate(dataset['target']):
        true_tag = dataset['target'][i]
        feature_vector = np.array([dataset['data'][i]])
        print("feature_vector: ", np.shape(feature_vector))
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
    logging.info('--------------------------------- \n')

# ------------------------------------------------------------------

# for subjects in range(1, 27):


for subject in range(1, 27):
    session = 5
    dataset = get_dataset(subject, session)
    model = open('EMD_ch14_HHT/EMD_ch14_HHT_ins40.sav', 'rb')
    clf = pickle.load(model)
    logging.info(" -------- Subject: {0} --------".format(subject))
    eval_model(dataset, clf)
