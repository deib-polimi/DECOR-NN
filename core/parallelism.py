import os
import shutil
from multiprocessing import Value, Event, Manager
from typing import List
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile
from torch.utils.data import DataLoader
from torch import nn
from core.SamplerUtils import DistributedSkipSampler
from torchsnapshot import Snapshot

from core.TrainingParams import TrainingParams

import ast

def read_file(filename):
    try:
        f = open(filename, "r")
        x = f.readline()
        value = ast.literal_eval(x)
        f.close()
        if not isinstance(value, list):
            raise ValueError
    except (OSError, SyntaxError, ValueError):
        value = None
    return value


class _ParallelGPUMode:
    train_object: TrainingParams
    devices: List[torch.device] | List[int]
    step: Value
    start_time: float

    def __init__(self, train_object, devices: List[torch.device] | List[int], progress_timeline: str, allocations_timeline: str, progress_value: Value, progress_event: Event, allocate_gpus_num: Value, allocated_gpus_num: Value):
        self.allocations_timeline = allocations_timeline
        self.progress_timeline = progress_timeline
        self.progress_value = progress_value
        self.progress_event = progress_event
        self.allocate_gpus_num = allocate_gpus_num
        self.allocated_gpus_num = allocated_gpus_num

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.train_object = train_object

        self.step = Value('i', 0)
        self.set_devices(devices, False)

    def set_devices(self, devices, update=True):

        temp_device = devices
        if all(isinstance(x, torch.device) for x in devices):
            if all(x.type == "cuda" for x in devices):
                try:
                    temp_device = list(map(lambda x: x.index, devices))
                except AttributeError:
                    temp_device = list(range(0, torch.cuda.device_count()))


        if all(isinstance(x, int) for x in temp_device):
            if update:
                self._write_allocation(temp_device)
            self.devices = temp_device
        else:
            raise TypeError("All devices must be a list of integers or torch.device")

    def _read_progress(self):
        with self.progress_value.get_lock():
            return self.progress_value.value

    def _read_devices(self):
        with self.allocate_gpus_num.get_lock():
            value = self.allocate_gpus_num.value
        return list(range(0, value))

    def start(self, epochs):
        pass

    def _write_allocation(self, new_gpus):
        with self.allocated_gpus_num.get_lock():
            self.allocated_gpus_num.value = len(new_gpus)

        alloc_time = time.monotonic()-self.start_time
        with open(self.allocations_timeline, "a+") as file:
            file.write(str(alloc_time) + "," + str(len(self.devices)) + '\n')
            file.write(str(alloc_time) + "," + str(len(new_gpus)) + '\n')

    def _setup_files(self):

        with open(self.progress_timeline, "w") as file:
            file.write("0,0\n")
        with open(self.allocations_timeline, "w") as file:
            file.write("0," + str(len(self.devices)) + '\n')
        with self.progress_value.get_lock():
            self.progress_value.value = 0
        self.progress_event.set()
        with self.allocated_gpus_num.get_lock():
            self.allocated_gpus_num.value = len(self.devices)

    def _end_files(self, epochs):
        end = time.monotonic() - self.start_time
        progress = self.train_object.get_num_batches() * epochs
        with open(self.allocations_timeline, "a+") as file:
            file.write(str(end) + "," + str(len(self.devices)) + '\n')
            file.write(str(end) + ",0")
        with open(self.progress_timeline, "a+") as file:
            file.write(str(end) + "," + str(progress))
        with self.progress_value.get_lock():
            self.progress_value.value += 1
        self.progress_event.set()


class DataParallel(_ParallelGPUMode):

    def __init__(self, train_object, devices: List[torch.device] | List[int], progress_timeline: str, allocations_timeline: str, progress_value: Value, progress_event: Event, allocate_gpus_num: Value, allocated_gpus_num: Value):
        super().__init__(train_object, devices, progress_timeline, allocations_timeline, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num)

    def start(self, epochs):

        if epochs < 1:
            raise ValueError("Expected positive number of epochs. Got: " + str(epochs))

        print("Starting Data Parallel.\n")
        self._setup_files()
        self.train_object.scale_lr(len(self.devices))
        model_with_devices = self._wrap()
        data_loader = self._get_data_loader()

        self.start_time = time.monotonic()

        epoch = 0
        while epoch < epochs:

            new_devices = self._train(model_with_devices, data_loader, epoch)

            if new_devices:
                self.set_devices(new_devices)
                self.train_object.model.load_state_dict(model_with_devices.module.state_dict())
                model_with_devices = self._wrap()
                self.train_object.scale_lr(len(self.devices))


            if self.step.value >= self.train_object.get_num_batches():
                self.step.value = 0
                data_loader = self._get_data_loader()
                epoch += 1

            skip = self.step.value * self.train_object.batch_size
            data_loader.sampler.set_skip(skip)

        self.train_object.model.load_state_dict(model_with_devices.module.state_dict())
        self._end_files(epochs)

    def _wrap(self):
        model_with_devices = nn.DataParallel(self.train_object.model, self.devices)
        model_with_devices.to(self._get_main_device())
        return model_with_devices

    def _get_data_loader(self):
        data_sampler = DistributedSkipSampler(dataset=self.train_object.train_dataset, num_replicas=1,
                                              rank=0)

        data_loader = DataLoader(dataset=self.train_object.train_dataset,
                                                  batch_size=self.train_object.batch_size*len(self.devices), shuffle=False,
                                                  num_workers=0, pin_memory=True, sampler=data_sampler)
        return data_loader

    def _train(self, model, data_loader, epoch):

        device = self._get_main_device()

        model.train()
        data_loader.sampler.set_epoch(epoch)
        for batch, (X, y) in enumerate(data_loader):

            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = self.train_object.loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            self.train_object.optimizer.step()
            self.train_object.optimizer.zero_grad()

            self._update_progress(epoch)

            devices = self._read_devices()
            if devices != self.devices:
                return devices

        return list()

    def _get_main_device(self):
        return torch.device((self.devices[0]))

    def _update_progress(self, epoch):

        self.step.value = min(self.step.value + len(self.devices), self.train_object.get_num_batches())
        progress = self.train_object.get_num_batches() * epoch + self.step.value

        with self.progress_value.get_lock():
            self.progress_value.value = progress
        self.progress_event.set()
        with open(self.progress_timeline, "a+") as file:
            file.write(str(time.monotonic() - self.start_time) + "," + str(progress) + '\n')



class DistributedDataParallel(_ParallelGPUMode):

    seed: int
    update: Value
    batch_per_process: Manager
    epoch_counter: Value
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    TIMES: str

    def __init__(self, train_object, devices: List[torch.device] | List[int], progress_timeline: str, allocations_timeline: str, progress_value: Value, progress_event: Event, allocate_gpus_num: Value, allocated_gpus_num: Value):
        super().__init__(train_object, devices, progress_timeline, allocations_timeline, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        self.update = Value('i', 0)
        self.synced_threads = Value('i', 0)
        self.epoch_counter = Value('i', 0)
        self.TIMES = os.path.join(os.path.dirname(__file__), "times.csv")
        self._init_times()


    def _init_times(self):
        self.file_time = Value('f', 0)

    def _end_time(self):
        with open(self.TIMES, "a+") as file:
            file.write("wrap_time,"+str(self.file_time.value)+"\n")

    def start(self, epochs):

        if epochs < 1:
            raise ValueError("Expected positive number of epochs. Got: " + str(epochs))

        self._setup_files()

        self.start_time = time.monotonic()
        skip = 0
        while self.epoch_counter.value < epochs:
            n_devices = len(self.devices)

            self.train_object.scale_lr(n_devices)

            mp.spawn(self._wrap, nprocs=n_devices, args=(epochs, skip), join=True)

            if self.update.value >= 1:

                devices = self._read_devices()

                if devices != self.devices:
                    self.set_devices(devices)
                self.update.value = 0

            skip = self.step.value * self.train_object.batch_size

        self._end_files(epochs)
        shutil.rmtree(self.CHECKPOINT_PATH)

        self._end_time()

    def _get_data_loader(self, world_size, rank):

        dist_sampler = DistributedSkipSampler(dataset=self.train_object.train_dataset, num_replicas=world_size, rank=rank)

        dist_loader = DataLoader(dataset=self.train_object.train_dataset,
                                                  batch_size=self.train_object.batch_size, shuffle=False,
                                                  num_workers=0, pin_memory=True, sampler=dist_sampler)
        return dist_loader

    def _wrap(self, rank, epochs, skip):

        torch.cuda.set_device(rank)
        world_size = len(self.devices)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method='env://')

        device = torch.device(self.devices[rank])
        model = self.train_object.model.to(device)
        model_with_devices = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        loader = self._get_data_loader(world_size, rank)
        self._train(rank, world_size, model_with_devices,  epochs, device, skip, loader)
        dist.barrier()
        dist.destroy_process_group()


    def _train(self, rank, world_size, model, max_epochs, device, skip, loader):
        app_state = {
            "model": model
        }
        if os.path.exists(self.CHECKPOINT_PATH):
            snapshot = Snapshot(path=self.CHECKPOINT_PATH)
            snapshot.restore(app_state=app_state)

        for epoch in range(self.epoch_counter.value, max_epochs):

            loader.sampler.set_skip(skip)
            loader.sampler.set_epoch(epoch)
            model.train()
            batch_per_process = 0
            with (model.join()):

                for batch, (X, y) in enumerate(loader):
                    X, y = X.to(device), y.to(device)
                    # Compute prediction error
                    pred = model(X)
                    loss = self.train_object.loss_fn(pred, y)
                    # Backpropagation
                    loss.backward()
                    self.train_object.optimizer.step()
                    self.train_object.optimizer.zero_grad()
                    batch_per_process += 1


                    if rank == 0:
                        self.step.value = min(self.step.value + world_size, self.train_object.get_num_batches())
                        if self.step.value == self.train_object.get_num_batches():
                            self.step.value = 0
                            self.epoch_counter.value += 1
                        self._update_progress(self.epoch_counter.value)

                        devices = self._read_devices()
                        if devices != self.devices and self.update.value == 0:
                            print(f"Found new devices {devices}.")
                            self.update.value = batch_per_process + 1

                    if batch_per_process == self.update.value:
                        break

            skip = 0
            if self.update.value >= 1:
                break
        self.save_state(app_state)




    def save_state(self, app_state):
        Snapshot.take(path=self.CHECKPOINT_PATH, app_state=app_state)

    def _update_progress(self, epoch):
        progress = self.train_object.get_num_batches() * epoch + self.step.value


        with self.progress_value.get_lock():
            self.progress_value.value = progress
        self.progress_event.set()
        with open(self.progress_timeline, "a+") as file:
            file.write(str(time.monotonic() - self.start_time) + "," + str(progress) + '\n')