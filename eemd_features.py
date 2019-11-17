from PyEMD import EEMD
from features import get_HHT

if __name__ == '__main__':

    def get_imfs_eemd(band):
        eemd = EEMD(trials=5)
        eIMFs = eemd(band, max_imf=4)
        return eIMFs[2:]


    def get_features_eemd(freq_bands, fs):
        features_vector = []
        for channel, bands in enumerate(freq_bands):
            for i, band in enumerate(bands):
                imfs = get_imfs_eemd(band)
                features_vector += get_HHT(imfs, fs)
                # features_vector += get_energy_values(imfs)
        return features_vector







