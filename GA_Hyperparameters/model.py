import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unittest


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self, NKERN, KERNEL, NFILT, HIDDEN, NGRUS, NOUT=4):
        super(NN, self).__init__()
        self.conv0 = nn.Conv2d(1, NFILT, kernel_size=(NKERN, KERNEL), padding=(0, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT, hidden_size=HIDDEN, num_layers=NGRUS, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(HIDDEN, NOUT)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        return x

