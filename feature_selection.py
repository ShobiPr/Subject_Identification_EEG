import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
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
ch_fs_instances = []
ch_tags_instances = []
for subject in range(1, 26):  # 26
    for session in range(1, 2):  # 4
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        instances = get_samples(_index, s_s_chs, sr)
        for f_instance in range(0, 10):
            instance = np.array(instances[f_instance, :, 1:-1]).transpose()
            ins14 = instance[[4, 5, 7, 9, 13, 15, 17, 23, 25, 33, 43, 51, 55, 56], :]
            ch_fs_instances.append(get_features_emd(ins14, sr)) # CHANNELS:8
            ch_tags_instances.append('subject_{0}'.format(subject))
dataset = {"data": ch_fs_instances, "target": ch_tags_instances}


for i, ii in enumerate(dataset['data']):
    color = "red" if dataset['target'][i] == "subject_1" else (
        "green" if dataset['target'][i] == "subject_2" else "blue")
    plt.plot(ii, color)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.grid(True)
plt.show()


dataTraining = dataset['data']
targetTraining = dataset['target']
stat_features = ['mean', 'var', 'std', 'kurtosis', 'skew', 'max', 'min', 'median']

model = ExtraTreesClassifier()
model.fit(dataTraining, targetTraining)
print(model.feature_importances_)

# plot graph of feature importance's - visualization
feat_importance = pd.Series(model.feature_importances_, index=stat_features)
feat_importance.nlargest(8).plot(kind='barh')
plt.savefig('feature_selection.png')