#! /usr/bin/env python3

import ast
import threading
import subprocess
import os
import time

from torch import multiprocessing as mp
from multiprocessing import Value, Event
import torch
from core import parallelism
from core.TrainingParams import TrainingParams

#Create global variables.
T_SAMPLE_SECONDS = 5.0
MIN_GPUS = 1
MAX_GPUS = torch.cuda.device_count()
K = 50
TI_SECONDS = 12
TOTAL_PROGRESS = 0
start_time = 0
deadline = 0
csi_old = (MAX_GPUS, MAX_GPUS)
gpus = MAX_GPUS
last_allocated = list(range(0, MAX_GPUS))
time_units = 0
spike_percentage = 0.995

#Calculate next allocation of GPUs
def next_allocation(progress, set_point, allocated_gpus_num):

    global csi_old

    csp = K * (set_point - (progress / TOTAL_PROGRESS))
    if last_allocated == allocated_devices(allocated_gpus_num):
        actual_csi = csi_old[1]
    else:
        actual_csi = csi_old[0]
    csi = actual_csi + csp * (T_SAMPLE_SECONDS / TI_SECONDS)
    cs = min(max(MIN_GPUS, csp + csi), MAX_GPUS)

    csi_new = cs - csp
    csi_old = (actual_csi, csi_new)
    cs = round(cs)
    return cs


#Commit changes to amount of allocated devices
def update(progress, allocate_gpus_num, allocated_gpus_num):

    global time_units
    global gpus

    time_units += T_SAMPLE_SECONDS
    set_point = time_units / deadline

    if set_point >= 1.0:
        next_gpu = gpus
        #If the controller is late, max out allocations only if the progress is lower than spike percentage. If the amount of missing progress is low then keep the same amount of allocations
        if progress/TOTAL_PROGRESS < spike_percentage:
            next_gpu = MAX_GPUS
    else:
        next_gpu = next_allocation(progress, set_point, allocated_gpus_num)

    if next_gpu != gpus:
        gpus = next_gpu
        try:
            allocate_gpus(gpus, allocate_gpus_num)
        except subprocess.CalledProcessError:
            print(f"Error in updating!")
    return

#Write allocations to variable, so that it can be detected by the parallelism module.
def allocate_gpus(num_gpus: int, allocate_gpus_num):

    global last_allocated
    last_allocated = list(range(0, num_gpus))

    with allocate_gpus_num.get_lock():
        allocate_gpus_num.value = num_gpus

    return last_allocated

#Read progress variable.
def read_progress(progress_value, progress_event):

    progress_event.wait()
    progress_event.clear()
    with progress_value.get_lock():
        value = progress_value.value

    return value

#Read devices currently being used by the parallelism module
def allocated_devices(allocated_gpus_num):

    with allocated_gpus_num.get_lock():
        return list(range(0, allocated_gpus_num.value))

#Repeatedly schedule update processes every T_SAMPLE_SECONDS
def schedule(is_done, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num):
    timer = threading.Timer(T_SAMPLE_SECONDS, schedule, (is_done, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num))
    timer.start()
    progress = read_progress(progress_value, progress_event)
    if progress <= TOTAL_PROGRESS:
        update(progress, allocate_gpus_num, allocated_gpus_num)
    else:
        timer.cancel()
        is_done.value = 1
        return


def start(train_object: TrainingParams, desired_deadline: float, alpha: float,
          epochs: int, progress_timeline: str, allocations_timeline: str, dl_change = False, distributed = True):

    global start_time
    global TOTAL_PROGRESS
    global deadline
    global csi_old
    global gpus
    global time_units


    time_units = 0
    TOTAL_PROGRESS = train_object.get_num_batches() * epochs
    deadline = desired_deadline * alpha
    csi_old = (MAX_GPUS, MAX_GPUS)
    gpus = MAX_GPUS
    is_done = Value('i', 0)
    progress_value = Value('i', 0)
    progress_event = Event()
    allocate_gpus_num = Value('i', 0)
    allocated_gpus_num = Value('i', 0)
    allocated_gpus = allocate_gpus(MAX_GPUS, allocate_gpus_num)

    if distributed:
        multi_gpu = parallelism.DistributedDataParallel(train_object, allocated_gpus, progress_timeline, allocations_timeline, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num)
    else:
        multi_gpu = parallelism.DataParallel(train_object, allocated_gpus, progress_timeline, allocations_timeline, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num)

    p = mp.Process(target=multi_gpu.start, args=(epochs,))
    p.start()

    #Training starts when multi_gpu initializes the progress at 0.
    read_progress(progress_value, progress_event)
    schedule(is_done, progress_value, progress_event, allocate_gpus_num, allocated_gpus_num)
    print(f"Starting Deadline Control at deadline {desired_deadline}s with alpha = {alpha}.")

    start_time = time.time()
    time.sleep(T_SAMPLE_SECONDS)

    #This section can be removed after the dynamic change is moved outside this module.
    #This is due to the fact that scheduling is already done by the schedule() function.
    while not bool(is_done.value):
        time.sleep(T_SAMPLE_SECONDS)
        #Dynamically change deadline. After prototyping this should be moved outside controller.
        if time_units / deadline >= 0.3 and dl_change:
            deadline = desired_deadline * 0.8
            print(f"Deadline changed to {deadline}\n")
            break
    if dl_change:
        while not bool(is_done.value):
            time.sleep(T_SAMPLE_SECONDS)


    p.join()
    end = time.time() - start_time
    print(f"Finished training after {end} seconds.")
    return end

