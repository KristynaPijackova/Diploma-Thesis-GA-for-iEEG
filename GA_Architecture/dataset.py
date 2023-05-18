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
    def __init__(self, path, csv_file):
        self.path = path
        self.csv_file = csv_file
        if self.path[-1] != '/':
            self.path += '/'
        self.df = pd.read_csv(self.path + self.csv_file)
        self.df['segment_id'] = self.df['segment_id'].apply(lambda x: self.path + x)
        self.NFFF = 200
        self.window = 'hann'
        self.nperseg = 32
        self.noverlap = 16
        self.nfft = 128

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

    def split_reviewer(self, reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        idx = self.df['reviewer_id'] != reviewer_id

        train.df = train.df[idx].reset_index(drop=True)
        valid.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train, valid

    def split_random(self, N_valid):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        train.df = train.df.iloc[N_valid:].reset_index(drop=True)
        valid.df = valid.df.iloc[:N_valid].reset_index(drop=True)
        return train, valid

    def cross_validation(self, N=10):
        datasets = [copy.deepcopy(self) for i in range(N)]
        kf = KFold(n_splits=N)
        for idx, (train_index, test_index) in enumerate(kf.split(self.df['category_id'].to_numpy())):
            datasets[idx].df = datasets[idx].iloc[test_index].reset_index(drop=True)

        return datasets

    def integrity_check(self):
        # iterate through dataset and check if all the files might be correctly loaded
        try:
            for i in tqdm.tqdm(range(len(self))):
                x = self.__getitem__(i)
        except Exception as exc:
            raise exc

    def remove_powerline_noise_class(self):
        self.df = self.df[self.df['category_id'] != 0]
        self.df['category_id'] = self.df['category_id'] - 1
        self.df = self.df.reset_index(drop=True)
        return self

#
# if __name__ == "__main__":
#     dataset_fnusa = Dataset('./dataset/test/*.mat').integrity_check()
#     dataset_mayo = Dataset('./dataset/train/*.mat').integrity_check()