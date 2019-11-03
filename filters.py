from __future__ import division
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
import warnings

warnings.filterwarnings("ignore")


def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'train/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    _header = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _header = _rows
            _header = _header.split(',')
    return np.array(channels), np.array(_header[1:-1])


def get_dataset():
    sr = 200
    s_s_chs, _header = get_subdataset()
    _index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
    instances = get_samples(_index, s_s_chs, sr)  # (60, 260, 59)
    instance = np.array(instances[1, :, 1:-1]).transpose()  # (57, 260)
    return instance


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
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


# Plots
def plot_frequency_response_low(b, a, fs, cutoff_lowpass):
    w, h = freqz(b, a, worN=8000)
    plt.subplot(4, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff_lowpass, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff_lowpass, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


def plot_frequency_response_high(b, a, fs, cutoff_highpass):
    w, h = freqz(b, a, worN=8000)
    plt.subplot(4, 1, 3)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff_highpass, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff_highpass, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Higpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


def plot_lowpass_highpass():
    # Filter requirements
    order = 6
    fs = 200.0  # sample rate, Hz
    cutoff_lowpass = 50.0  # desired cutoff frequency of the filter, Hz
    cutoff_highpass = 0.01

    T = 1.3  # seconds
    n = int(T * fs)  # tot n_samples
    t = np.linspace(0, T, n)

    dataset = get_dataset()
    data = dataset[0]

    # -------------------------------------------------------------------------------

    b_l, a_l = butter(order, cutoff_lowpass / (0.5 * fs), btype='low', analog=False)
    plot_frequency_response_low(b_l, a_l, fs, cutoff_lowpass)

    y_lowpass = butter_lowpass_filter(data, cutoff_lowpass, fs, order)

    plt.subplot(4, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y_lowpass, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()


    b_h, a_h = signal.butter(order, cutoff_highpass / (0.5 * fs), btype='high', analog=False)
    plot_frequency_response_high(b_h, a_h, fs, cutoff_highpass)

    y_highpass = butter_highpass_filter(data, cutoff_highpass, fs, order=5)

    plt.subplot(4, 1, 4)
    plt.plot(t, y_lowpass, 'b-', label='lowpass filtered data')
    plt.plot(t, y_highpass, 'r-', linewidth=2, label='highpass + lowpass filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.show()


def plot_lowpass():
    fs = 200.0  # sample rate, Hz
    cutoff_lowpass = 50.0  # desired cutoff frequency of the filter, Hz
    T = 1.3  # seconds
    n = int(T * fs)  # tot n_samples
    t = np.linspace(0, T, n)

    dataset = get_dataset()

    for i in range(0,5):
        y_lowpass = butter_lowpass_filter(dataset[0], cutoff_lowpass, fs, order=3)
        plt.subplot(5, 1, i+1)
        plt.plot(t, dataset[0], 'b-', label='data')
        plt.plot(t, y_lowpass, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    plt.show()


def plot_frequency_bands():
    order = 6
    fs = 200.0  # sample rate, Hz
    lowcut = 50.0  # desired cutoff frequency of the filter, Hz
    highcut = 0.01

    T = 1.3  # seconds
    n = int(T * fs)  # tot n_samples
    t = np.linspace(0, T, n)

    dataset = get_dataset()
    data = dataset[0]

    plt.subplot(5, 1, 1)
    delta = delta_wave(data, fs)
    plt.plot(t, delta, label='delta')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.subplot(5,1,2)
    theta = theta_wave(data, fs)
    plt.plot(t, theta, label='theta')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.subplot(5,1,3)
    alpha = alpha_wave(data, fs)
    plt.plot(t, alpha, label='alpha')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.subplot(5,1,4)
    beta = beta_wave(data, fs)
    plt.plot(t, beta, label='beta')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.subplot(5,1,5)
    gamma = gamma_wave(data, fs)
    plt.plot(t, gamma, label='gamma')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


"""
order = 6
fs = 200.0  # sample rate, Hz
lowcut = 50.0  # desired cutoff frequency of the filter, Hz
highcut = 0.01

T = 1.3  # seconds
n = int(T * fs)  # tot n_samples
t = np.linspace(0, T, n)

dataset = get_dataset()
data = dataset[0]


bandpasset = butter_bandpass_filter(data, 0.01, 50.0, fs, order=order)
plt.subplot(2,1,1)
plt.plot(t, bandpasset, label='bandpassed')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

low = butter_lowpass_filter(data, 50.0, fs, order=order)
low_and_high = butter_highpass_filter(low, 0.01, fs, order=order)
plt.subplot(2,1,2)
plt.plot(t, low_and_high, label='low_and_high')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.show()


"""