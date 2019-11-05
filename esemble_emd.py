from PyEMD import EEMD, Visualisation
import numpy as np
import math
import matplotlib.pyplot as plt
from dataset import get_dataset


def get_imfs_eemd(signal):
    signal = np.array(signal)
    components = EEMD()(signal)
    imfs, res = components[:-1], components[-1]
    return imfs[1:4]


def get_imfs(ch_freq_bands):
    ch_imfs = []
    for channel, bands in enumerate(ch_freq_bands):
        eIMFs = []
        for band, signal in enumerate(bands):
            eIMFs.append(get_imfs_eemd(signal))
        ch_imfs.append(eIMFs)
    return ch_imfs


def plot_imfs(freq_bands, ch_imfs):
    name_band = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for channel, bands in enumerate(ch_imfs):
        for band, imfs in enumerate(bands):
            nIMFs = imfs.shape[0]
            # band_signal = freq_bands[channel][band]
            plt.subplot(nIMFs + 1,1,1)
            plt.plot(freq_bands[channel][band])
            plt.title(name_band[band])

            for n in range(nIMFs):
                plt.subplot(nIMFs + 1, 1, n + 2)
                plt.plot(imfs[n])
                plt.ylabel("IMF %i" % (n + 1))
                plt.locator_params(axis='y', nbins=5)
            plt.show()





