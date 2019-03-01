from torch.utils import data
from tqdm import tqdm

import numpy as np
import torch
import os

LABELS = {
    'A-major': 0, 'Amajor': 1, 'Aminor': 2,
    'B-major': 3, 'B-minor': 4, 'Bmajor': 5, 'Bminor': 6,
    'C#major': 7, 'C#minor': 8, 'Cmajor': 9, 'Cminor': 10,
    'Dmajor': 11, 'Dminor': 12,
    'E-major': 13, 'E-minor': 14, 'Emajor': 15, 'Eminor': 16,
    'F#major': 17, 'F#minor': 18, 'Fmajor': 19, 'Fminor': 20,
    'G#minor': 21, 'Gmajor': 22, 'Gminor': 23
}

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

        label                  = np.zeros((len(LABELS)))
        label[LABELS[data[1]]] = 1
        label                  = torch.from_numpy(label).unsqueeze(0)
        imgs                   = torch.from_numpy(np.load(data[0])).unsqueeze(0)

        return imgs, label
