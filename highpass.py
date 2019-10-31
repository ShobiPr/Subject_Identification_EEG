from __future__ import division

from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
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


dataset = get_dataset()

"""
def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({
        'data' : y_sine} ,index=t_sine)
    return result
"""

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



data = dataset[0]
fs = 200
cutoff = 0.01

filtered_sine = butter_highpass_filter(data, cutoff, fs, order=6)

b, a = butter_highpass(cutoff, fs, order=6)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Highpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

T = 1.3  # seconds
n = int(T * fs)  # tot n_samples
t = np.linspace(0, T, n)


plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, filtered_sine, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.show()

"""
plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(data)), data)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(data)),data)
plt.plot(range(len(filtered_sine)), filtered_sine)
plt.title('filtered signal')
plt.show()

# Filter requirements
order = 6
fs = 200.0  # sample rate, Hz
cutoff = 50.0 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_highpass_filter(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()




T = 1.3  # seconds
n = int(T * fs)  # tot n_samples
t = np.linspace(0, T, n)

data = dataset[5]

# Filter the data, and plot both the original and filtered signals.
y = butter_highpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

"""

