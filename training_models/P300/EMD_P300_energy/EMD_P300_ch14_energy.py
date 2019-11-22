from dataset import get_subdataset, get_samples
from features import get_imfs_emd, get_energy_values
from preprocessing import preprocessing
from classefiers import random_forest, decision_tree, knn, SVM, naive_bayes, selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_P300_ch14_energy.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def get_features_emd(instance):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_emd(channel)
        if len(imfs) > 1:
            features_vector += get_energy_values(imfs)
    return features_vector


def main():
    logging.info(" ***** CHANNELS:14, FEATURES: energy ***** \n")
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
        for subject in range(1, 27):  # 26
            for session in range(1, 5):  # 4
                s_s_chs = get_subdataset(subject, session)
                _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
                instances = get_samples(_index, s_s_chs, sr)
                for f_instance in range(0, ins):
                    instance = preprocessing(lowcut, highcut, f_instance, instances, order, sr)
                    ins14 = instance[[4, 5, 7, 9, 13, 15, 17, 23, 25, 33, 43, 51, 55, 56], :]
                    ch_fs_instances.append(get_features_emd(ins14))  # CHANNELS:14
                    ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'EMD_P300_ch14_energy_ins%02d.sav' % (ins)
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()