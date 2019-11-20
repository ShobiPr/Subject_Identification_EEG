import numpy as np
import matplotlib.pyplot as plt

from dataset import get_subdataset, get_samples
from features import get_features_emd


sr = 200
session = 1
ch_fs_instances = []
ch_tags_instances = []
sub_ins_m = []
for subject in range(1, 27):  # 26
    s_s_chs = get_subdataset(subject, session)
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)
    for f_instance in range(0, 15):
        instance = np.array(instances[f_instance, :, 1:-1]).transpose()
        ch_fs_instances.append(get_features_emd(instance, sr))
    sub_ins_m.append(np.mean(ch_fs_instances, axis=0))
    ch_tags_instances.append('subject_{0}'.format(subject))
    print("Subjects {0}".format(subject))
dataset = {"data": sub_ins_m, "target": ch_tags_instances}


for i, ii in enumerate(dataset['data']):
    plt.plot(ii, linewidth=0.8)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.title('Energy features')
plt.show()
# plt.savefig('features_stat.png')

