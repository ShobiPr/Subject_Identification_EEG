# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pylab as plt
import math
from preprosessing import preprocessing
import warnings

from pyhht import EMD
from pyhht.visualization import plot_imfs

warnings.filterwarnings("ignore")


def get_imfs_emd(signal):
    # decomposer_signal = EMD(signal, fixe=100, n_imfs=2)
    decomposer_signal = EMD(signal)
    imfs = decomposer_signal.decompose()

    nIMFs = len(imfs)

    print("nIMFS: ", nIMFs)
    for n in range(nIMFs):
        plt.subplot(nIMFs + 4, 1, n + 2)
        plt.plot(imfs[n])
        plt.ylabel("IMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)
    plt.show()



