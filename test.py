from esemble_emd import get_imfs_eemd, plot_imfs
from dataset import get_dataset
from filters import frequency_bands, plot_freq_band
import matplotlib.pyplot as plt
from hilbert_transform import hilbert_transform
from PyEMD import EEMD
import numpy as np


dataset = get_dataset()  # (57, 260) aka (channels, samples)

# --------------------------------------------

sr = 200
freq_bands = []  # (1, [5 (delta, theta, alpha, beta, gamma), 260])
for channel, samples in enumerate(dataset):
    if channel < 1:
        freq_bands.append(frequency_bands(samples, sr))

"OK"

# --------------------------------------------

def get_imfs(signal):
    signal = np.array(signal)
    components = EEMD()(signal)
    imfs, res = components[:-1], components[-1]
    return imfs

delta = freq_bands[0][0]
eIMFs = get_imfs(delta)

print("eIMFS: ", np.shape(eIMFs))

# Plot results
neIMFs = eIMFs.shape[0]
plt.subplot(neIMFs + 1, 1, 1)
plt.plot(delta, 'r')
plt.ylabel("delta")

for n in range(neIMFs):
    plt.subplot(neIMFs + 1, 1, n + 2)
    plt.plot(eIMFs[n], 'g')
    plt.ylabel("eIMF %i" % (n + 1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()


"""
ch_imfs = []  # (1 [channels], 5 [freq bands], [3 [imfs], 260])
for channel, bands in enumerate(freq_bands):  # bands: (5, 260)
    IMFs = []
    for band, signal in enumerate(bands):
        IMFs.append(get_imfs_eemd(signal))
    ch_imfs.append(IMFs)
"""


# plot_imfs(freq_bands, ch_imfs)

"NOT OK BUT AINT GOT TIME"

# --------------------------------------------

# ch_instFreq, ch_instAmp = hilbert_transform(ch_imfs, sr)

# print("instFreq: ", np.shape(ch_instFreq))
# print("instAmp: ", np.shape(ch_instAmp))