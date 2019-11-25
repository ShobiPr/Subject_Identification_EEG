from __future__ import division, print_function

from preprocessing import preprocessing_P300
from features import get_energy_values
from PyEMD import EEMD

from dataset_P300 import  get_subdataset, get_samples


def get_imfs_eemd(instance):
    eemd = EEMD(trials=15)
    eIMFs = eemd.eemd(instance, max_imf=4)
    return eIMFs


def get_features_eemd(instance):
    features_vector = []
    for i, channel in enumerate(instance):
        imfs = get_imfs_eemd(channel)
        features_vector += get_energy_values(imfs)
    return features_vector


def main():
    sr = 200
    ch_fs_instances = []
    ch_tags_instances = []
    ch = [7, 15, 25, 33, 43, 51, 55, 56]
    for subject in range(1, 2):  # 26
        for session in range(1, 2):  # 4
            s_s_chs = get_subdataset(subject, session)
            _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
            instances = get_samples(_index, s_s_chs, sr)
            for f_instance in range(0, 10):
                print("Subject: ", subject)
                instance = preprocessing_P300(instances, f_instance, sr, ch)
                ch_fs_instances.append(get_features_eemd(instance))  # CHANNELS: 8
                ch_tags_instances.append('subject_{0}'.format(subject))
    return {"data": ch_fs_instances, "target": ch_tags_instances}


if __name__ == "__main__":
    dataset = main()