# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import math
from scipy import signal
from filters import butter_bandpass_filter
import warnings
import os
import pywt
from pywt import WaveletPacket, WaveletPacket2D
import pywt.data


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


def test():

    # dataset = get_dataset()
    # data = dataset[0]

    x = np.linspace(0, 1, num=512)
    data = np.sin(250 * np.pi * x ** 2)

    wp = WaveletPacket(data, 'bior2.2', maxlevel=5)

    fig = plt.figure()
    plt.set_cmap('bone')
    ax = fig.add_subplot(wp.maxlevel + 1, 1, 1)
    ax.plot(data, 'k')
    ax.set_xlim(0, len(data) - 1)
    ax.set_title("Wavelet packet coefficients")

    for level in range(1, wp.maxlevel + 1):
        ax = fig.add_subplot(wp.maxlevel + 1, 1, level + 1)
        nodes = wp.get_level(level, "freq")  # Collects nodes from the given level of decomposition, Specifies nodes order (frequecy)
        nodes.reverse()
        labels = [n.path for n in nodes]  # Path string defining position of the node in the decomposition tree
        values = -abs(np.array([n.data for n in nodes]))
        ax.imshow(values, interpolation='nearest', aspect='auto')
        ax.set_yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.show()


test()

def wp_visualize_coeffs_distribution():
    ecg = pywt.data.ecg()

    wp = WaveletPacket(ecg, 'sym5', maxlevel=4)

    fig = plt.figure()
    plt.set_cmap('bone')
    ax = fig.add_subplot(wp.maxlevel + 1, 1, 1)
    ax.plot(ecg, 'k')
    ax.set_xlim(0, len(ecg) - 1)
    ax.set_title("Wavelet packet coefficients")

    for level in range(1, wp.maxlevel + 1):
        ax = fig.add_subplot(wp.maxlevel + 1, 1, level + 1)
        nodes = wp.get_level(level, "freq")  # Collects nodes from the given level of decomposition, Specifies nodes order (frequecy)
        nodes.reverse()
        labels = [n.path for n in nodes]  # Path string defining position of the node in the decomposition tree
        values = -abs(np.array([n.data for n in nodes]))
        ax.imshow(values, interpolation='nearest', aspect='auto')
        ax.set_yticks(np.arange(len(labels) - 0.5, -0.5, -1), labels)
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.show()


def wp_2d():
    arr = pywt.data.aero()

    wp2 = WaveletPacket2D(arr, 'db2', 'symmetric', maxlevel=2)

    # Show original figure
    plt.imshow(arr, interpolation="nearest", cmap=plt.cm.gray)

    path = ['d', 'v', 'h', 'a']

    # Show level 1 nodes
    fig = plt.figure()
    for i, p2 in enumerate(path):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(np.sqrt(np.abs(wp2[p2].data)), origin='image',
                  interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(p2)

    # Show level 2 nodes
    for p1 in path:
        fig = plt.figure()
        for i, p2 in enumerate(path):
            ax = fig.add_subplot(2, 2, i + 1)
            p1p2 = p1 + p2
            ax.imshow(np.sqrt(np.abs(wp2[p1p2].data)), origin='image',
                      interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(p1p2)

    fig = plt.figure()
    i = 1
    for row in wp2.get_level(2, 'freq'):
        for node in row:
            ax = fig.add_subplot(len(row), len(row), i)
            ax.set_title("%s=(%s row, %s col)" % (
                         (node.path,) + wp2.expand_2d_path(node.path)))
            ax.imshow(np.sqrt(np.abs(node.data)), origin='image',
                      interpolation="nearest", cmap=plt.cm.gray)
            i += 1

    plt.show()


def wp_scalogram():
    x = np.linspace(0, 1, num=512)
    data = np.sin(250 * np.pi * x**2)

    wavelet = 'db2'
    level = 4
    order = "freq"  # other option is "normal"
    interpolation = 'nearest'
    cmap = plt.cm.cool

    # Construct wavelet packet
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    labels = [n.path for n in nodes]
    values = np.array([n.data for n in nodes], 'd')
    values = abs(values)

    # Show signal and wavelet packet coefficients
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title("linchirp signal")
    ax.plot(x, data, 'b')
    ax.set_xlim(0, x[-1])

    ax = fig.add_subplot(2, 1, 2)
    ax.set_title("Wavelet packet coefficients at level %d" % level)
    ax.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
              origin="lower", extent=[0, 1, 0, len(values)])
    ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels)

    # Show spectrogram and wavelet packet coefficients
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211)
    ax2.specgram(data, NFFT=64, noverlap=32, Fs=2, cmap=cmap,
                 interpolation='bilinear')
    ax2.set_title("Spectrogram of signal")
    ax3 = fig2.add_subplot(212)
    ax3.imshow(values, origin='upper', extent=[-1, 1, -1, 1],
               interpolation='nearest')
    ax3.set_title("Wavelet packet coefficients")


    plt.show()

