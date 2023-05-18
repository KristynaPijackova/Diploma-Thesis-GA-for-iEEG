import numpy as np
import random
import string
from datetime import datetime


class Solution:
"""This class monitors parameters relevant for our results and updates them."""
    def __init__(self, x: dict):
        self.pop_size = None
        self.parameters = x
        self.score = np.inf  # assign infinite value for the initial weights of each parameter
        self.f1 = np.inf
        self.auroc = np.inf
        self.auprc = np.inf
        self.auroc_all = np.inf
        self.auprc_all = np.inf
        self.f1_all = np.inf
        self.loss = np.inf
        self.epoch = np.inf
        self.id = ''.join(random.choice(string.ascii_letters) for i in range(32))
        self.assigned = False
        self.assigned_timestamp = None
        self.finished = False
        self.duration = None

    def __str__(self):
        return f"{self.id} {self.assigned_timestamp} {self.finished} {self.score} {self.parameters}"

    def __repr__(self):
        return f"{self.id} {self.assigned_timestamp} {self.finished} {self.score} {self.parameters}"

    def __eq__(self, other):
        if self.parameters == other.parameters:
            return True
        return False

    def assign(self):
        self.assigned = True
        self.assigned_timestamp = datetime.timestamp(datetime.now())
        self.finished = False
