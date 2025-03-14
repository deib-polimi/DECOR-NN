import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

def dp_run(model, loss_fn, optimizer, dataset, epochs, batch_size):
    x = time.monotonic()
    device = torch.device("cuda:0")
    n_devices = torch.cuda.device_count()
    dp_model = DP(model)
    dp_model.to(device)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size*n_devices, shuffle=False,
                                         num_workers=0, pin_memory=True)
    #We avoid scaling learning rate as we only care for training time.

    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = dp_model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return time.monotonic() - x

def ddp_run(model, loss_fn, optimizer, dataset, epochs, batch_size):
    x = time.monotonic()
    n_devices = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(ddp_train, nprocs=n_devices, args=(n_devices, model, loss_fn, optimizer, dataset, epochs, batch_size), join=True)
    return time.monotonic() - x

def cleanup():
    dist.destroy_process_group()

def ddp_train(rank, world_size, model, loss_fn, optimizer, dataset, epochs, batch_size):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"Running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    model = model.to(rank)
    device = torch.device(rank)
    ddp_model = DP(model, device_ids=[rank])
    dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=0, pin_memory=True, sampler=dist_sampler)
    model.train()
    # We avoid scaling learning rate as we only care for training time.
    for epoch in range(epochs):
        loader.sampler.set_epoch(epoch)
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = ddp_model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    cleanup()