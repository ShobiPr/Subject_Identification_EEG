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



"""
dataset = get_dataset()
signal = dataset[0]
modes = get_imfs_eemd(signal)  # (3, 260) aka. (3 x imf, samples)
"""

"""
    nIMFs = imfs.shape[0]
    eIMFSshape = eIMFS.shape[0]

    plt.subplot(eIMFSshape + 1, 1, 1)
    plt.plot(signal, 'r')

    for n in range(eIMFSshape):
        plt.subplot(eIMFSshape + 1, 1, n + 2)
        plt.plot(eIMFS[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
"""

"""
    # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(signal, 'r')

    for n in range(1, 4):
        plt.subplot(4, 1, n + 1)
        plt.plot(imfs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()
"""







