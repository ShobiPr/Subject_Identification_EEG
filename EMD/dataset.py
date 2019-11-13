from PyEMD import EEMD, Visualisation
import numpy as np
import math
import logging

def get_samples(_index, s_s_chs, sr, _size=1.3):
    instances = []
    for _ind in _index:
        instances.append(s_s_chs[_ind:int(math.ceil(_ind + (_size * sr)))][:])
    return np.array(instances)


def get_subdataset(_S=1, Sess=1):
    _file = 'P300/Data_S%02d_Sess%02d.csv' % (_S, Sess)
    _f = open(_file).readlines()
    channels = []
    _header = []
    for i, _rows in enumerate(_f):
        if i > 0:
            channels.append(eval(_rows))
        else:
            _header = _rows
    return np.array(channels)


logging.basicConfig(filename='test.log',
                    level=logging.INFO,
                    format='%(levelname)s:%(message)s')

logging.info(" ***** CHANNELS:8, FEATURES: ENERGY ***** \n")
INSTANCES = [10, 20, 40, 60]
for ins in INSTANCES:

    logging.info(" -------- Instance: {0} --------".format(ins))
