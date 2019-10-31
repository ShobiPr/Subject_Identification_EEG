from __future__ import division
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
from scipy import fftpack
from skimage import util
import warnings


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
    # y = lfilter(b, a, data)
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
    y = lfilter(b, a, data)
    return y


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


"""
fs = 200.0
lowcut = 1.0
highcut = 50.0


# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [3, 6, 9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=100)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.show()


T = 1.3  # seconds
n = int(T * fs)  # tot n_samples
t = np.linspace(0, T, n, endpoint=False)
data = get_dataset()
x = data[0]
plt.figure(2)
plt.clf()
plt.plot(t, x, label='Noisy signal')

y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
plt.plot(t, y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.show()
"""

