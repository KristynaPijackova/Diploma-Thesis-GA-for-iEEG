##################################################################################
#   Example code 2                                                               #
#   Train CNN GRU model on dataset from one hospital and test on second hospital #
#   This should be done for each hospital -> cross validation of results         #
##################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os
#from model import *
from modelKristyna import *
from dataset import *
from statistics import *

from tqdm import tqdm
from datetime import date

def run_model(WINDOW, NPERSEG, NOVERLAP, NFFT, NFILT, KERNEL, HIDDEN, NGRUS, BATCH, LR, L2):
    # DATA INITIALIZATION
    dataset_train = Dataset('./train/', 'train_small_1_fnusa.csv', WINDOW, NPERSEG, NOVERLAP, NFFT)
    dataset_test = Dataset('./test/', 'valid_fnusa.csv', WINDOW, NPERSEG, NOVERLAP, NFFT)
    kern = dataset_train.__getitem__(0)[0].shape[1]
    NWORKERS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    TRAIN = DataLoader(dataset=dataset_train,
                       batch_size=BATCH,
                       shuffle=True,
                       drop_last=False,
                       num_workers=NWORKERS)

    TEST = DataLoader(dataset=dataset_test,
                      batch_size=BATCH,
                      shuffle=True,
                      drop_last=False,
                      num_workers=NWORKERS)

    # MODEL INITIALIZATION
    model = NN(NKERN=kern, KERNEL=KERNEL, NFILT=NFILT, HIDDEN=HIDDEN, NGRUS=NGRUS, NOUT=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    loss = nn.CrossEntropyLoss()
    statistics = Statistics()

    writer = SummaryWriter()

    # TRAIN
    for epoch in range(10):
        model.train()
        for i, (x, t) in enumerate(tqdm(TRAIN)):
            optimizer.zero_grad()
            x = x.to(DEVICE).float()
            t = t.to(DEVICE).long()
            y = model(x)[:, -1, :]
            J = loss(input=y, target=t)
            J.backward()
            optimizer.step()


        model.eval()
        for i, (x, t) in enumerate(tqdm(TEST)):
            x = x.to(DEVICE).float()
            t = t.to(DEVICE).long()
            y = model(x)[:, -1, :]

            statistics.append(target=t, logits=y)
        kappa = statistics.evaluate()
        writer.add_scalar('Kappa/valid', kappa, epoch)
    writer.close()
    return kappa