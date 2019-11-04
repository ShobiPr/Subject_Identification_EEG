from scipy.signal import hilbert
import numpy as np


def hilbert_transform(bands, fs):
    ch_instFreq = []
    ch_instAmp = []
    for band, imfs in enumerate(bands):
        hs = (hilbert([imfs[i] for i in range(len(imfs))]))
        ch_instAmp = np.abs(hs)  # Calculate amplitude
        omega_s = np.unwrap(np.angle(hs))  # Unwrap phase
        ch_instFreq = (np.diff(omega_s) / (2 * np.pi / fs))  # Calculate instantaneous frequency
    return ch_instFreq, ch_instAmp

