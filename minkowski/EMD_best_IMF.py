from temp.dataset_P300 import get_subdataset, get_samples
from temp.features import get_features_emd
from temp.preprocessing import preprocessing_P300

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='minkowski.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def main():
    logging.info(" ***** Minkowski distance ***** \n")
    sr = 200

    ch_fs_instances = []
    ch_tags_instances = []
    for subject in range(1, 2):  # 26
        for session in range(1, 2):  # 4
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(0, 60):
                logging.info("Instance: {0}".format(f_instance + 1))
                instance = preprocessing_P300(instances, f_instance, sr)
                ch_fs_instances.append(get_features_emd(instance, sr))
        print("subject {0}: ".format(subject))


if __name__ == '__main__':
    main()