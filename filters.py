from __future__ import division
from scipy.signal import freqz, iirnotch, filtfilt, butter
import warnings

warnings.filterwarnings("ignore")


def notch_filter(s, sr, f0=50, Q=10.0):
    b, a = iirnotch(f0, Q, sr)
    return filtfilt(b, a, s)


def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fs, lowcut=0.5, highcut=70.0, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=6):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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