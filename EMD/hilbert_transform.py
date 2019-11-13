from scipy.signal import hilbert
import numpy as np


def hilbert_transform(bands, fs):
    ch_instFreq = []
    ch_instAmp = []
    for band, imfs in enumerate(bands):
        hs = (hilbert([imfs[i] for i in range(len(imfs))]))
        ch_instAmp.append(np.abs(hs))  # Calculate amplitude
        omega_s = np.unwrap(np.angle(hs))  # Unwrap phase
        ch_instFreq.append((np.diff(omega_s) / (2 * np.pi / fs)))  # Calculate instantaneous frequency
    return ch_instFreq, ch_instAmp


def hilbert_transform_prodo(band, fs):
    band_instFreq = []
    band_instAmp = []
    for m, mode in enumerate(band):
        hs = hilbert(mode)
        band_instAmp.append(np.abs(hs))  # Calculate amplitude
        omega_s = np.unwrap(np.angle(hs))  # Unwrap phase
        band_instFreq.append((np.diff(omega_s) / (2 * np.pi / fs)))  # Calculate instantaneous frequency
    return band_instFreq, band_instAmp