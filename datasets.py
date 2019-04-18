from __future__ import print_function
import errno
import os
import torch
import torch.utils.data as data
from PIL import Image
from random import randint
import numpy as np

class MnistLayoutDataset(data.Dataset):
    def __init__(self, path='data/processed/training.pt', element_num=128, gt_thresh=200):
        super(MnistLayoutDataset, self).__init__()
        self.train_data = torch.load(path)[0]
        self.element_num = element_num
        self.gt_thresh = gt_thresh

    def __getitem__(self, index):
        img = self.train_data[index]
        gt_thold = self.gt_thresh        
        gt_values = []

        for id, i in enumerate(img):
            for jd, j in enumerate(i):
                if j >= gt_thold:
                    gt_values.append(torch.FloatTensor([1, np.float32(2*id +1)/56, np.float32(2*jd +1)/56]))
        
        grph_elements = []
        for _ in range(self.element_num):
            ridx = randint(0, len(gt_values)-1)
            grph_elements.append(gt_values[ridx])
        
        # MNIST layout elements format : [1, x, y]
        return torch.stack(grph_elements)

    def __len__(self):
        return len(self.train_data)
    
