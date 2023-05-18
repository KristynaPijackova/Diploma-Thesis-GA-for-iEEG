import torch
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from kymatio.torch import Scattering1D
import scipy.signal as signal

class Dataset:
"""Dataset class for the wavelet scattering transformation of the data."""
    def __init__(self, path, file_name):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        self.file_name = file_name
        self.df = pd.read_csv(self.path + self.file_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sid = self.df.iloc[item]['segment_id']
        target = self.df.iloc[item]['category_id']
        data = sio.loadmat(self.path + '{}'.format(sid))['data']
        # Normalization
        # data = stats.zscore(data, axis=1)
        # Zero-padding
        zero_pad = int((2 ** 14 - data.shape[-1]) / 2)
        data_pad = np.pad(data[-1], (zero_pad, zero_pad), mode='constant')
        return data, data_pad, target


def first_order_scattering(data, DEVICE, J, Q, T=2**14):
    print('wst')
    scattering = Scattering1D(J, data.shape[-1], Q)
    scattering.to(DEVICE)
    meta = scattering.meta()
    order1 = np.where(meta['order'] == 1)
    Sx = scattering(data)
    data = Sx[:, np.array(order1), :]

    return data[:, :, :, :]

