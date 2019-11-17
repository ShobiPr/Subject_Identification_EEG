import numpy as np
from dataset import get_subdataset, get_samples
from filters import get_frequency_bands
from eemd_features import get_features_eemd
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EEMD.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

def main():
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 11):  # 10
        for session in range(1, 2):  # 1
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(0, 2):  # 2
                instance = np.array(instances[f_instance, :, 1:-1]).transpose()
                # ch8_ins = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
                ins = instance[[55], :]
                freq_bands = get_frequency_bands(ins, sr) # 1 channel
                ch_fs_instances.append(get_features_eemd(freq_bands, sr))
                ch_tags_instances.append('subject_{0}'.format(subject))
    dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

    dataTraining = dataset['data']
    targetTraining = dataset['target']
    result = selector(dataTraining, targetTraining)

    logging.info("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))


if __name__ == "__main__":
    main()
