from scipy.signal import hilbert
import numpy as np


def hilbert_transform(ch_imfs, fs):
    ch_instFreq = []
    ch_instAmp = []
    for channel, bands in enumerate(ch_imfs):
        instFreq = []
        instAmp = []
        for band, imfs in enumerate(bands):
            hs = (hilbert([imfs[i] for i in range(len(imfs))]))
            instAmp.append(np.abs(hs))  # Calculate amplitude
            omega_s = np.unwrap(np.angle(hs))  # Unwrap phase
            instFreq.append(np.diff(omega_s) / (2 * np.pi / fs))  # Calculate instantaneous frequency
            print("instFreq: ", np.shape(instFreq))
        ch_instFreq.append(instFreq)
        ch_instAmp.append(instAmp)
    return ch_instFreq, ch_instAmp

