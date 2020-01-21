from dataset_P300 import get_subdataset, get_samples
from features import get_features_emd
import numpy as np
from classefiers import selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='P300_EMD_ch7_energy_hht_nopre.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def main():
    logging.info(" ***** EMD, CHANNELS:7, FEATURES: energy + hht ***** ")
    logging.info("--------------no preprocessing-------------- \n")
    INSTANCES = [10, 20]
    sr = 200
    ch = [1, 42, 46, 51, 52, 54, 55]
    # ch =  ch = [0, 1, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 36, 36, 38, 40,
    # 41, 42, 44, 46, 48, 50, 51, 52, 53, 54, 55]  32 channels
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
                    if f_instance not in [15, 21, 23, 45]:
                        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
                        ins7 = instance[ch, :]
                        ch_fs_instances.append(get_features_emd(ins7, sr))  # CHANNELS: 14
                        ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'P300_EMD_ch7_energy_hht_nopre_ins%02d.sav' % ins
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()