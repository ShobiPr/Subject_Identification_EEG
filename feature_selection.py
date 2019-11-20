import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

from dataset import get_subdataset, get_samples
from features import get_features_emd

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename='EMD_P300_ch8_stat.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')

logging.info(" ***** CHANNELS:8, FEATURES: STAT ***** \n")


sr = 200
cutoff = 50.0
order = 6
session = 1
ch_fs_instances = []
ch_tags_instances = []
ins = []
sub_ins_m = []
for subject in range(1, 27):  # 26
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, 60):
        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
        ch_fs_instances.append(get_features_emd(instance, sr))
    sub_ins_m.append(np.mean(ch_fs_instances, axis=0))
    ch_tags_instances.append('subject_{0}'.format(subject))
dataset = {"data": sub_ins_m, "target": ch_tags_instances}


for i, ii in enumerate(dataset['data']):
    """color = "red" if dataset['target'][i] == "subject_1" else (
        "green" if dataset['target'][i] == "subject_2" else "blue")"""
    plt.plot(ii, linewidth=0.8, label='Subject {0}'.format(i))
plt.xlabel('Feature')
plt.ylabel('Value')
#plt.grid(True)
plt.legend()
plt.title('Statistical features')
plt.savefig('features_stat.png')

