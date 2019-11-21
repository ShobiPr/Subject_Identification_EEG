import numpy as np
import matplotlib.pyplot as plt

from dataset import get_subdataset, get_samples
from features import get_HHT, get_energy_values, get_fractal_values, get_statistics_values, get_imfs_emd


""" Feature selection on statistical, energy, freactal, HHT-based features
    All channels used (57) with 60 instances each - calculate the mean
    For channels with more than 1 IMF, features are calculated on (first two) IMFs and creates n 
    features for each IMF -> 1 ch = n*2 features """


def get_features_emd(instance, fs):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_emd(channel)
        if len(imfs) > 1:
            features_vector += get_statistics_values(imfs)
            # features_vector += get_energy_values
            # features_vector += get_fractal_values
            # features_vector += get_HHT
    return features_vector


sr = 200
session = 1
ch_fs_instances = []
ch_tags_instances = []
sub_ins_m = []
for subject in range(2, 27):  # 26
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, 60):
        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
        ch_fs_instances.append(get_features_emd(instance, sr))
    sub_ins_m.append(np.mean(ch_fs_instances, axis=0))
    print("Subjects {0} done".format(subject))


for i, ii in enumerate(sub_ins_m):
    plt.plot(ii, linewidth=0.8)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.title('Statistical features')
plt.savefig('features_stat.svg', format='svg')
plt.show()


