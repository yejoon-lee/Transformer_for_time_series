import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_loader(input, target, batch_size, shuffle=True):
    '''Make dataloder for pyTorch.
    '''
    dset = TensorDataset(torch.Tensor(input), torch.Tensor(target))
    dloader = DataLoader(dset, shuffle=shuffle, batch_size=batch_size)
    return dloader


class EarlyStopping:
    def __init__(self, patience, verbose=0, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None: # first __call__
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta: # when model didn't improve
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # when model improved
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss