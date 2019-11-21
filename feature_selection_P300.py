import numpy as np
import matplotlib.pyplot as plt

from dataset import get_subdataset, get_samples
from features import get_features_emd


""" Feature selection on statistical, energy, freactal, HHT-based features
    All channels used (57) with 60 instances each - calculate the mean
    For channels with more than 1 IMF, features are calculated on (first two) IMFs and creates n 
    features for each IMF -> 1 ch = n*2 features """


sr = 200
session = 1
ch_fs_instances = []
ch_tags_instances = []
sub_ins_m = []
for subject in range(1, 11):  # 26
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)

    for f_instance in range(0, 15):
        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
        ins8 = instance[[7, 15, 25, 33, 43, 51, 55, 56], :]
        ch_fs_instances.append(get_features_emd(ins8, sr))

    m = np.mean(ch_fs_instances, axis=0)
    sub_ins_m.append(m)
    print("Subjects {0} done".format(subject))
dataset = {"data": sub_ins_m, "target": ch_tags_instances}


for i, ii in enumerate(dataset['data']):
    plt.plot(ii, linewidth=0.8)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.title('HHT features')
plt.savefig('stat_features_stat.svg', format='svg')
plt.show()


