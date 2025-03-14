from math import ceil
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset

#Class that is used to persist model data between threads.
class TrainingParams:

    model: torch.nn.Module
    train_dataset: Dataset
    optimizer: torch.optim.Optimizer
    loss_fn: _Loss
    batch_size: int
    scaling: Callable

    def __init__(self, model, train_dataset, loss_fn, optimizer, batch_size, scaling):

        self.model = model
        self.train_dataset = train_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.base_lr = self._get_lr()
        self.batch_size = batch_size
        self.scaling = scaling

    def get_num_batches(self):
        return ceil(len(self.train_dataset)/self.batch_size)

    def _get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def scale_lr(self, n_devices):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scaling(n_devices) * self.base_lr