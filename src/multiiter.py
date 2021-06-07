import random
import itertools
from torch.utils.data import IterableDataset

class MultiIter(IterableDataset):
    def __init__(self, I1, I2, paug=0.5):
        self.I1 = itertools.cycle(I1)
        self.I2 = itertools.cycle(I2)
        self.paug = paug

    def __iter__(self):
        return self

    def __len__(self):
        return min(len(self.I1) / self.paug, len(self.I2) / (1-self.paug))

    def __next__(self):
        if random.random() < self.paug:
            return next(self.I1)
        else:
            return next(self.I2)
