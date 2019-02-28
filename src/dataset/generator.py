from torch.utils import data
from tqdm import tqdm

import numpy as np
import torch
import os

class Dataset(data.Dataset):
    def __init__(self, data_dir):
        self.data = self._read_files(data_dir)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def _read_files(self, data_dir):
        labels  = os.listdir(data_dir)
        folders = [os.path.join(data_dir, label) for label in labels]

        data = []
        pbar = tqdm(enumerate(labels), total=len(labels), desc='Reading files per class')
        for i, label in pbar:
            folder = folders[i]
            paths  = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
            ]

            data += [(path, label) for path in paths]

        return data

    def __getitem__(self, idx):
        data = self.data[idx]

        label = data[1]
        imgs  = torch.from_numpy(np.load(data[0])).unsqueeze(0)

        return imgs, label
