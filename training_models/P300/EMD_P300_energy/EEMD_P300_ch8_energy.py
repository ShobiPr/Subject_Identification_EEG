from dataset_P300 import get_subdataset, get_samples
import numpy as np
from features import get_imfs_eemd, get_energy_values
from preprocessing.preprocessing import preprocessing_P300
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EEMD_P300_ch8_energy.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def get_features_emd(instance):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_eemd(channel)
        if len(imfs) > 1:
            features_vector += get_energy_values(imfs)
    return features_vector


def main():
    logging.info(" ***** CHANNELS:8, FEATURES: energy ***** \n")
    logging.info(" ---------- With preprocessing ---------- \n")
    # INSTANCES = [10, 20, 40, 60]
    INSTANCES = [10]
    sr = 200
    for ins in INSTANCES:
        logging.info(" -------- Instance: {0} --------".format(ins))
        ch_fs_instances = []
        ch_tags_instances = []
        for subject in range(1, 4):  # 26
            for session in range(1, 2):  # 4
                s_s_chs = get_subdataset(subject, session)
                _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
                instances = get_samples(_index, s_s_chs, sr)
                for f_instance in range(0, ins):
                    instance = preprocessing_P300(instances, f_instance, sr)
                    ins8 = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
                    ch_fs_instances.append(get_features_emd(ins8))  # CHANNELS: 8
                    ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'EEMD_P300_ch8_energy_ins%02d.sav' % (ins)
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()