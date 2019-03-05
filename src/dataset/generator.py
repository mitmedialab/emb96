from dataset.midi import load_img, SAMPLE_PER_MEASURE, MEASURES, NOTES
from torch.autograd import Variable
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
        self.labels  = sorted(os.listdir(data_dir))
        folders      = [
            os.path.join(data_dir, label)
            for label in self.labels
        ]

        pbar = tqdm(
            enumerate(self.labels),
            total = len(self.labels),
            desc  = 'Reading files per class'
        )
        data = []
        for i, label in pbar:
            folder = folders[i]
            paths  = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
            ]

            data += [(path, label) for path in paths]

        return data

    def _load_imgs(self, path):
        measures = load_img(path)
        if measures.shape[1] < MEASURES * SAMPLE_PER_MEASURE:
            _measures = np.zeros((NOTES, MEASURES * SAMPLE_PER_MEASURE))
            _measures[:, :measures.shape[1]] = measures
            measures                         = _measures

        start    = np.random.randint(0, measures.shape[1] - (MEASURES * SAMPLE_PER_MEASURE) + 1)
        measures = measures[:, start:start + (MEASURES * SAMPLE_PER_MEASURE)]
        data     =  np.array([
            measures[:, i:i + SAMPLE_PER_MEASURE]
            for i in range(0, measures.shape[1], SAMPLE_PER_MEASURE)
        ], dtype=np.uint8)
        return data.reshape(MEASURES, NOTES, SAMPLE_PER_MEASURE)

    def __getitem__(self, idx):
        data = self.data[idx]

        label_idx        = self.labels.index(data[1])
        label            = np.zeros((len(self.labels)))
        label[label_idx] = 1

        label = torch.from_numpy(label).float()
        imgs  = torch.from_numpy(self._load_imgs(data[0])).float()

        return (imgs, label)
