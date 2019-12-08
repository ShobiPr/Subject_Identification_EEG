from temp.dataset_P300 import get_subdataset, get_samples
from temp.features import get_features_emd
from preprocessing import preprocessing_P300
from classefiers import selector

import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD3_P300_ch8_HHT.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def main():
    logging.info(" ***** EMD3, CHANNELS:8, FEATURES: HHT ***** \n")
    INSTANCES = [10, 20, 40, 60]
    sr = 200
    ch = [7, 15, 25, 33, 43, 51, 55, 56]
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
                    instance = preprocessing_P300(instances, f_instance, sr, ch)
                    ch_fs_instances.append(get_features_emd(instance, sr))  # CHANNELS: 8
                    ch_tags_instances.append('subject_{0}'.format(subject))
        dataset = {"data": ch_fs_instances, "target": ch_tags_instances}

        dataTraining = dataset['data']
        targetTraining = dataset['target']
        result = selector(dataTraining, targetTraining)

        logging.info("Best classifier {0} with accuracy {1} \n".format(result['classifier'], result['accuracy']))

        # saving the model
        model_name = 'EMD3_P300_ch8_HHT_ins%02d.sav' % ins
        pickle.dump(result["model"], open(model_name, 'wb'))


if __name__ == '__main__':
    main()