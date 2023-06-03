import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
import sys


class LocationDatasetTrain(Dataset):
    def __init__(self, x_train, y_train, seq_len) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.seq_len = seq_len


    def __len__(self):
        return len(self.x_train) - self.seq_len
    

    def __getitem__(self, index):
        return self.x_train[index : index + self.seq_len], self.y_train[index + self.seq_len]


class LocationDatasetTest(Dataset):
    def __init__(self, x_test, y_test, seq_len) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.seq_len = seq_len


    def __len__(self):
        return len(self.x_test) - self.seq_len


    def __getitem__(self, index):
        return self.x_test[index : index + self.seq_len], self.y_test[index + self.seq_len]   

