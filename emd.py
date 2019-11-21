from dataset import get_subdataset, get_samples
from preprocessing import preprocessing
import numpy as np
from features import get_features_emd
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_P300_ch8_stat.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')

logging.info(" ***** CHANNELS:8, FEATURES: STAT ***** \n")
logging.info(" -------- Instance: {0} --------".format(ins))
sr = 200
cutoff = 50.0
order = 6
ch_fs_instances = []
ch_tags_instances = []
for subject in range(1, 5):  # 26
    for session in range(1, 2):  # 4
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        instances = get_samples(_index, s_s_chs, sr)
        for f_instance in range(0, ins):  # INSTANCES = [10, 20, 30, 40]
            instance = np.array(instances[f_instance, :, 1:-1]).transpose()
            #ins14 = instance[[4, 5, 7, 9, 13, 15, 17, 23, 25, 33, 43, 51, 55, 56], :]
            #ins8 = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
            ch_fs_instances.append(get_features_emd(instance, sr))
            ch_tags_instances.append('subject_{0}'.format(subject))
    print("subject {0}: ".format(subject))
dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

dataTraining = dataset['data']
targetTraining = dataset['target']
result = selector(dataTraining, targetTraining)

logging.info("Best classifier {0} with accuracy {1}".format(result['classifier'], result['accuracy']))

# saving the model
model_name = 'EMD_ch8_stat_ins%02d.sav' % (ins)
pickle.dump(result["model"], open(model_name, 'wb'))