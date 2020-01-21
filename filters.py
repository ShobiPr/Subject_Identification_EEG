from __future__ import division
from scipy.signal import freqz, iirnotch, filtfilt, butter
import matplotlib.pyplot as plt
import math
import numpy as np
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
    return butter_bandpass_filter(signal, fs, 0.5, 4.0, order=4)


def theta_wave(signal, fs):
    return butter_bandpass_filter(signal, fs, 4.0, 8.0, order=5)


def alpha_wave(signal, fs):
    return butter_bandpass_filter(signal, fs, 8.0, 12.0, order=5)


def beta_wave(signal, fs):
    return butter_bandpass_filter(signal, fs, 12.0, 30.0, order=5)


def gamma_wave(signal, fs):
    return butter_highpass_filter(signal, 30.0, fs, order=5)


def frequency_bands(channel, fs):
    freq_band = [
        delta_wave(channel, fs),
        theta_wave(channel, fs),
        alpha_wave(channel, fs),
        beta_wave(channel, fs),
        gamma_wave(channel, fs)
    ]
    return freq_band


def plot_freq_bands(freq_band):
    name_wave = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for channel, waves in enumerate(freq_band):
        for i in range(len(waves)):
            plt.subplot(len(waves), 1, i + 1)
            plt.plot(waves[i])
            plt.ylabel("{}".format(name_wave[i]))
            # plt.ylabel("wave %i" % (i + 1))
            plt.locator_params(axis='y', nbins=5)
        plt.show()


def get_frequency_bands(ins, fs):  # (56, 260)
    ch_freq_bands = []
    for channel, samples in enumerate(ins):
        ch_freq_bands.append(frequency_bands(samples, fs))
    return ch_freq_bands


def preprocessing_P300(instances, f_instance, fs, lowcut=0.5, highcut=70.0, order=7):
    """ Notch filter, f0 = 50, Q=10.0
        bandpass : [0.5-70.0]Hz order:6 """
    instance = np.array(instances[f_instance, :, 1:-1]).transpose()
    filtered_instance = []
    for i, channel in enumerate(instance):
        notch_f = notch_filter(channel, fs, f0=50.0, Q=20.0)
        filtered_instance.append(butter_bandpass_filter(notch_f, fs, lowcut, highcut, order))
    return np.array(filtered_instance)

# ---------------------------------------------------------------------------

def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'P300/train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _headers = _rows
    return np.array(channels)


def get_signals_P300():
    sr = 200
    session = 1
    data = []
    for subject in range(1, 2):  # 26
        s_s_chs = get_subdataset(subject, session)
        _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
        instances = get_samples(_index, s_s_chs, sr)
        for f_instance in range(0, 1):
            instance = preprocessing_P300(instances, f_instance, sr)
            return instance
            # fb = get_frequency_bands(instance, sr)
            # plot_freq_bands(fb)
# ---------------------------------------------------------------------------

def lp():

    fs = 200.0  # sample rate, Hz

    b, a = butter_highpass(30.0, fs, order=3)
    w, h = freqz(b, a, worN=2000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    #plt.plot(12.0, 0.5 * np.sqrt(2), 'ko')
    plt.plot(30.0, 0.5 * np.sqrt(2), 'ko')
    #plt.axvline(12.0, color='k')
    plt.axvline(30.0, color='k')
    #plt.xlim(0, 12.0 * fs)
    plt.xlim(0, 30.0 * fs)
    plt.title("processed Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    data = get_signals_P300()
    signal = data[5]
    y = gamma_wave(signal, fs)

    plt.subplot(2, 1, 2)
    plt.plot(signal, 'b-', label='data')
    plt.plot(y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.grid()

    plt.subplots_adjust(hspace=0.35)
    plt.show()


"""
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
"""
"""
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    # dataset = get_signals_P300()
    dataset = get_signals_resting()
    data = dataset[0]
    T = 1.3
    n = T * fs
    t = np.linspace(0, T, 165, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)
"""


