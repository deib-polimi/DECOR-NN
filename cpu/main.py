#! /usr/bin/env python3
import os
import re
import random
import torch
import torchvision
import torch.nn as nn
import sys
from torch.utils.data import Dataset
import multiprocessing
import math
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time


class RandomDataset(Dataset):
    def __init__(self, values, labels):
        super(RandomDataset, self).__init__()
        self.values = values
        self.labels = labels

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


# Get the amount of channels required for a model. While this number is usually 3, it allows for dynamic retrieval.
def get_channel_depth(model):
    try:
        model(torch.empty(1, 0, 1, 1))
        search = 0
    except RuntimeError as e:
        search = re.search('(\\d+)[^\\d]+channels', str(e))
    channels = int(search.group(1))
    return channels


def create_random_dataset(model_str, model, samples):
    channels = get_channel_depth(model)
    # This is model dependent, most models have this as a minimum size. Inception v3 has minimum size 299x299
    height = 224
    width = 224
    values = torch.rand(samples, channels, height, width)

    metadata = torchvision.models.get_model_weights(model_str).DEFAULT.meta
    tot_categories = len(metadata["categories"])
    labels = torch.randint(tot_categories, (samples,)).tolist()

    return RandomDataset(values, labels)


def cpu_training(model, num_batches, epochs, batch_size, progress_file=None):
    train_model = torchvision.models.get_model(model)
    dataset = create_random_dataset(model, train_model, num_batches * batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.001)

    x = time.monotonic()
    device = "cpu"
    loader = DataLoader(dataset, batch_size=batch_size)

    train_model.train()
    if progress_file is not None:
        with open(progress_file, "w") as file:
            file.write("0")
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = train_model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if progress_file is not None:
                with open(progress_file, "w") as file:
                    file.write(str(epoch * num_batches + batch + 1))
    return time.monotonic() - x


if __name__ == "__main__":
    model = sys.argv[1]
    num_batches = int(sys.argv[2])
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    progress_file = os.path.join(os.path.dirname(__file__), "results", sys.argv[5])
    time = cpu_training(model, num_batches, epochs, batch_size, progress_file)