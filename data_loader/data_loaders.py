from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import pandas as pd
import numpy as np


class dataloader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = dataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        filelist = ['train_image_data_0.feather', 'train_image_data_1.feather',
                    'train_image_data_2.feather', 'train_image_data_3.feather']
        self.data = pd.concat([pd.read_feather(self.data_dir+filelist[0]), pd.read_feather(self.data_dir+filelist[1]),
                               pd.read_feather(self.data_dir+filelist[2]), pd.read_feather(self.data_dir+filelist[3])], axis=0).iloc[:, 1:].to_numpy().astype('uint8').reshape([-1, 1, 137, 236])
        print('Read the data completely!')
        self.target = pd.read_csv(self.data_dir+'train.csv')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]/255
        target1 = self.target.iat[idx, 1]
        target2 = self.target.iat[idx, 2]
        target3 = self.target.iat[idx, 3]
        return {'data': torch.from_numpy(data), 'target': [torch.tensor(target1), torch.tensor(target2), torch.tensor(target3)]}
