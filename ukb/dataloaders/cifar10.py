from __future__ import print_function, division
import os
import logging
import numpy as np
import pandas as pd
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


logger = logging.getLogger(__name__)


class CIFAR10(Dataset):
    """
    """
    def __init__(self, data_root, split="train", transform=None, num_samples=25000):
        self.dataset   = torchvision.datasets.CIFAR10(data_root, train=split in ['train','dev'], download=True)
        self.transform = transform

        if split == "train":
            self.data = self._load_images(self.dataset.train_data[0:num_samples])
            self.labels = [y for y in self.dataset.train_labels[0:num_samples]]
        elif split == "dev":
            self.data = self._load_images(self.dataset.train_data[50000 - num_samples:])
            self.labels = [y for y in self.dataset.train_labels[50000 - num_samples:]]
        else:
            self.data = self._load_images(self.dataset.test_data)[0:num_samples]
            self.labels = [y for y in self.dataset.test_labels][0:num_samples]

        self.dataset = None

    def _load_images(self, array):
        imgs = []
        for x in array:
            x = np.array(x).astype(np.float32)
            x = np.array([x[..., i] for i in range(x.shape[-1])])
            imgs.append(x)
        return imgs

    def summary(self):
        return "Instances: {}".format(len(self))

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.transform:
            raise NotImplemented()

        x,y = self.data[idx], self.labels[idx]
        return x, y

