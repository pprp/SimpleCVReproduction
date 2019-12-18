import os
import random

import numpy as np
import torch
from torchvision import datasets


class TripleFolder(datasets.ImageFolder):
    def __init__(self, data_root, transforms):
        super(TripleFolder, self).__init__(data_root, transforms)
        self.targets = np.asarray([s[1] for s in self.samples])
    
    def get_pos_sample(self, target, index):
        