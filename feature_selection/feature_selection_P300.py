import numpy as np
import matplotlib.pyplot as plt

from temp.preprocessing import preprocessing_P300
from temp.dataset_P300 import get_subdataset, get_samples
from temp.features import get_features_emd


sr = 200
session = 1
ch_fs_instances = []
ch_tags_instances = []
sub_ins_m = []
ch = [54, 55]
for subject in range(1, 10):  # 26

    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)

    for f_instance in range(0, 60):
        instance = preprocessing_P300(instances, f_instance, sr, ch)
        ch_fs_instances.append(get_features_emd(instance, sr))  # CHANNELS: 8

    m = np.mean(ch_fs_instances, axis=0)
    sub_ins_m.append(m)
    print("Subjects {0} done".format(subject))
dataset = {"data": sub_ins_m, "target": ch_tags_instances}


for i, ii in enumerate(dataset['data']):
    plt.plot(ii, linewidth=0.8)
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.grid(True)
# plt.tight_layout()
plt.suptitle('Feature selection')
plt.savefig('feature selection.svg', format='svg')
plt.show()


