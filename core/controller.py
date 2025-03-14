#! /usr/bin/env python3

import ast
import threading
import subprocess
import os
import time

from torch import multiprocessing as mp
from multiprocessing import Value
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
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "progress.txt")
DEVICE_FILE = os.path.join(os.path.dirname(__file__), "devices.txt")
ALLOCATED_DEVICE_FILE = os.path.join(os.path.dirname(__file__), "allocated_devices.txt")

#Calculate next allocation of GPUs
def next_allocation(progress, set_point):

    global csi_old

    csp = K * (set_point - (progress / TOTAL_PROGRESS))
    if last_allocated == allocated_devices():
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
def update(progress):

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
        next_gpu = next_allocation(progress, set_point)

    if next_gpu != gpus:
        gpus = next_gpu
        try:
            allocate_gpus(gpus)
        except subprocess.CalledProcessError:
            print(f"Error in updating!")
    return

#Write allocations to file, so that it can be detected by the parallelism module.
def allocate_gpus(num_gpus: int):

    global last_allocated
    last_allocated = list(range(0, num_gpus))

    with open(DEVICE_FILE, "w") as file:
        file.write(str(last_allocated))

    return last_allocated

#Read progress file.
def read_progress():

    value = -1
    while value < 0:
        try:
            with open(PROGRESS_FILE, "r") as file:
                value = int(file.readline())
        except (ValueError, FileNotFoundError):
            value = -1
            time.sleep(0.1)

    return value

#Read devices currently being used by the parallelism module
def allocated_devices():

    value = 0
    while not isinstance(value, list):
        try:
            with open(ALLOCATED_DEVICE_FILE, "r") as file:
                value = ast.literal_eval(file.readline())
        except (OSError, SyntaxError, ValueError):
            value = None
            time.sleep(0.1)
    return value

#Repeatedly schedule update processes every T_SAMPLE_SECONDS
def schedule(is_done):
    timer = threading.Timer(T_SAMPLE_SECONDS, schedule, (is_done,))
    timer.start()
    progress = read_progress()
    if progress <= TOTAL_PROGRESS:
        update(progress)
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
    allocated_gpus = allocate_gpus(MAX_GPUS)

    if distributed:
        multi_gpu = parallelism.DistributedDataParallel(train_object, allocated_gpus, progress_timeline, allocations_timeline, PROGRESS_FILE, DEVICE_FILE, ALLOCATED_DEVICE_FILE)
    else:
        multi_gpu = parallelism.DataParallel(train_object, allocated_gpus, progress_timeline, allocations_timeline,
                                                        PROGRESS_FILE, DEVICE_FILE, ALLOCATED_DEVICE_FILE)

    p = mp.Process(target=multi_gpu.start, args=(epochs,))
    p.start()

    #Training starts when multi_gpu creates a file with progress 0.
    read_progress()
    schedule(is_done)
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
    os.remove(PROGRESS_FILE)
    os.remove(DEVICE_FILE)
    return end

