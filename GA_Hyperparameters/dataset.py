import pandas as pd
import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm
import glob
from sklearn.model_selection import KFold


class Dataset:
    def __init__(self, path, csv_file, window, nperseg, noverlap, nfft):
        self.path = path
        self.csv_file = csv_file
        if self.path[-1] != '/':
            self.path += '/'
        self.df = pd.read_csv(self.path + self.csv_file)
        self.df['segment_id'] = self.df['segment_id'].apply(lambda x: self.path + x)

        self.NFFF = 200
        self.B, self.A = signal.butter(3, 500 / (5000 / 2), 'low')

        window_opt = [('tukey', 0.5), 'bartlett']
        self.window = window_opt[window]
        self.nperseg = 2 ** nperseg
        self.noverlap = self.nperseg // 2 ** noverlap
        if nperseg + nfft > 10:
            self.nfft = 2 ** 10
        else:
            self.nfft = 2 ** (nperseg + nfft)

    def __len__(self):
        return len(self.df)

    def __add__(self, other):
        out = copy.deepcopy(self)
        out.df = pd.concat([self.df, other.df])
        out.df = out.df.reset_index(drop=True)
        return out

    def keep_N_per_class(self, N):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        for i in range(4):
            idx = self.df[self.df.category_id == i].index[N:]
            self.df.loc[idx, 'category_id'] = 4
        stats = self.df['category_id'].value_counts()
        self.df = self.df[self.df['category_id'] != 4]
        stop = 1

    def __getitem__(self, item):
        sid = self.df.iloc[item]['segment_id']
        target = int(self.df.iloc[item]['category_id'])
        data = sio.loadmat(sid)['data']

        # data = signal.filtfilt(self.B,self.A,data)
        # data = signal.resample_poly(data,up=1,down=5)

        f, t, data = signal.spectrogram(data[0, :], window=self.window,
                                        fs=5000, nperseg=self.nperseg,
                                        noverlap=self.noverlap, nfft=self.nfft)
        # fs=1000, nperseg=256, noverlap=128, nfft=1024

        data = data[:self.NFFF, :]
        # data0 = stats.zscore(data,axis=0) ### check axis
        # data1 = stats.zscore(data,axis=1)
        # data = np.stack([data0,data1],axis=0)
        data = stats.zscore(data, axis=1)
        # data = (data - data.min())/(data.max()-data.min())
        data = np.expand_dims(data, axis=0)
        data = np.nan_to_num(data)

        return data, target

