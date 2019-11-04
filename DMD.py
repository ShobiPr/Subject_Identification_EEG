import __future__
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from features import get_features
from preprosessing import preprocessing
import warnings

warnings.filterwarnings("ignore")

from pydmd import DMD


def f1(x, t):
    return 1./np.cosh(x+3)*np.exp(2.3j*t)


def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)


x = np.linspace(-5, 5, 128)
t = np.linspace(0, 4*np.pi, 256)

xgrid, tgrid = np.meshgrid(x, t)

x1 = f1(xgrid, tgrid)
x2 = f2(xgrid, tgrid)
X = x1 + x2


titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data = [x1, x2, X]

fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
plt.show()

"""
dmd = DMD(svd_rank=2)
dmd.fit(X.T)

for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamic')
plt.show()
"""


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


sr = 200
s_s_chs, _header = get_subdataset()
_index = [i + 1 for i, d in enumerate(s_s_chs[:, -1]) if d == 1]
instances = get_samples(_index, s_s_chs, sr)
instance = np.array(instances[1, :, 1:-1]).transpose()

x1 = np.linspace(0, 1.3, 260)
print("x1: ", np.shape(x1))

dmd = DMD(svd_rank=2)
X = instance[0]
print("X = instance [0]: ", np.shape(instance[0]))
dmd.fit(X.T)

for mode in dmd.modes.T:
    print("mode: ", np.shape(mode))
    plt.plot(x1, mode.real)
    plt.title('Modes')
plt.show()