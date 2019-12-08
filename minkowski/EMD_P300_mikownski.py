from temp.dataset_P300 import get_subdataset, get_samples
import numpy as np
from temp.features import get_features_emd

import logging
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_resting_energy.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


sr = 200
cutoff = 50.0
order = 4
ch_fs_instances = []
ch_tags_instances = []
for subject in range(1, 2):  # 26
    for session in range(1, 2):  # 4
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        instances = get_samples(_index, s_s_chs, sr)
        for f_instance in range(0, 60):

            logging.info("Instance: ", (f_instance+1))
            instance = np.array(instances[f_instance, :, 1:-1]).transpose()
            ch_fs_instances.append(get_features_emd(instance, sr))
    print("subject {0}: ".format(subject))

