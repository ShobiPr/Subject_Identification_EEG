from __future__ import division
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
from dataset import get_dataset
import warnings

warnings.filterwarnings("ignore")


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


def low_and_high_vs_bandpass():
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
    plt.subplot(2, 1, 1)
    plt.plot(t, bandpasset, label='bandpassed')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    low = butter_lowpass_filter(data, 50.0, fs, order=order)
    low_and_high = butter_highpass_filter(low, 0.01, fs, order=order)
    plt.subplot(2, 1, 2)
    plt.plot(t, low_and_high, label='low_and_high')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()


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


def plot_bandpass():
    fs = 200.0  # sample rate, Hz
    lowcut = 0.01  # desired cutoff frequency of the filter, Hz
    highcut = 50.0
    order = 6
    T = 1.3  # seconds
    n = int(T * fs)  # tot n_samples
    t = np.linspace(0, T, n)

    dataset = get_dataset()

    for i in range(0,5):
        y_bandpass = butter_bandpass_filter(dataset[i], lowcut, highcut, fs, order=order)
        plt.subplot(5, 1, i+1)
        plt.plot(t, dataset[i], 'b-', label='data')
        plt.plot(t, y_bandpass, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
    plt.show()


def frequency_bands(signal, fs):
    freq_band = [
        delta_wave(signal, fs),
        theta_wave(signal, fs),
        alpha_wave(signal, fs),
        beta_wave(signal, fs),
        gamma_wave(signal, fs)
    ]
    return freq_band


# (5, [5 (delta, theta, alpha, beta, gamma), 260])
def plot_freq_band(freq_band):
    name_wave = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    for channel, waves in enumerate(freq_band):
        for i in range(len(waves)):
            plt.subplot(len(waves), 1, i + 1)
            plt.plot(waves[i])
            plt.ylabel("{}".format(name_wave[i]))
            # plt.ylabel("wave %i" % (i + 1))
            plt.locator_params(axis='y', nbins=5)
        plt.show()