from __future__ import division
from scipy.signal import butter, lfilter

from scipy import signal
import warnings

warnings.filterwarnings("ignore")


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

"""
def butter_bandpass(lowcut, highcut, sr, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')

def butter_bandpass_filter(signal, lowcut, highcut, sr, order=4):
    b, a = butter_bandpass_filter(lowcut, highcut, sr, order)
    y = lfilter(b, a, signal)
    return y
"""

def delta_wave(signal, fs):
    return butter_bandpass_filter(signal, 0.5, 4.0, fs)


def theta_wave(signal, fs):
    return butter_bandpass_filter(signal, 4.0, 8.0, fs)


def alpha_wave(signal, fs):
    return butter_bandpass_filter(signal, 8.0, 12.0, fs)


def beta_wave(signal, fs):
    return butter_bandpass_filter(signal, 12.0, 30.0, fs)


def gamma_wave(signal, fs):
    return butter_highpass_filter(signal, 30.0, fs)


def frequency_bands(channel, fs):
    freq_band = [
        delta_wave(channel, fs),
        theta_wave(channel, fs),
        alpha_wave(channel, fs),
        beta_wave(channel, fs),
        gamma_wave(channel, fs)
    ]
    return freq_band


"""
def get_frequency_bands(ins, fs):  # (56, 260)
    ch_freq_bands = []
    for channel, samples in enumerate(ins):
        ch_freq_bands.append(frequency_bands(samples, fs))
    return ch_freq_bands



def get_frequency_bands(instances, f_instance, ch_start, ch_stop, fs):  # (56, 260)
    ins = np.array(instances[f_instance, :, 1:-1]).transpose()
    ch_freq_bands = []
    for channel, samples in enumerate(ins):
        if ch_start <= channel <= ch_stop:
            ch_freq_bands.append(frequency_bands(samples, fs))
    return ch_freq_bands

"""