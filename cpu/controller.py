#! /usr/bin/env python3

import logging
import threading
import subprocess
import os
import multiprocessing as mp
import time
import math
from multiprocessing import Value

T_SAMPLE_SECONDS = 1
MIN_CORES = 1
MAX_CORES = float(mp.cpu_count())
K = 50
TI_SECONDS = 12
TOTAL_PROGRESS = 0
start_time = 0
deadline = 0
csi_old = MAX_CORES
cores = MAX_CORES
container_name = 'container'
CORE_QUANTUM = 0.05
QUANTUM_DIGITS = -int(math.floor(math.log10(CORE_QUANTUM)))
spike_percentage = 0.99
results_dir = os.path.join(os.path.dirname(__file__), "results")
PROGRESS_FILE = 0
ALLOCATIONS_FILE = 0
time_units = 0
logger = logging.getLogger(__name__)
PROGRESS_TIMELINE = 0


def next_allocation(progress, set_point):
    global csi_old

    csp = K * (set_point - (progress / TOTAL_PROGRESS))
    csi = csi_old + csp * (T_SAMPLE_SECONDS / TI_SECONDS)
    cs = min(max(MIN_CORES, csp + csi), MAX_CORES)

    # Originally cs = min(max(MIN_CORES, csp+csi), (TOTAL_PROGRESS-progress), MAX_CORES)

    csi_old = cs - csp
    cs = round(round(cs / CORE_QUANTUM) * CORE_QUANTUM, QUANTUM_DIGITS)
    return cs


def update(progress):
    global time_units
    global cores

    time_units += T_SAMPLE_SECONDS
    set_point = time_units / deadline

    if set_point >= 1.0:
        next_core = cores
        if progress / TOTAL_PROGRESS < spike_percentage:
            next_core = MAX_CORES
    else:
        next_core = next_allocation(progress, set_point)

    if next_core != cores:
        old_core = cores
        cores = next_core

        try:
            cpu_period = 100000
            subprocess.run('docker update --cpu-quota="' + str(int(cores * cpu_period)) + '" ' + container_name,
                           check=True,
                           shell=True, capture_output=True)

            alloc_time = time.time() - start_time
            with open(ALLOCATIONS_FILE, "a+") as file:
                file.write(str(alloc_time) + "," + str(old_core) + "\n")
                file.write(str(alloc_time) + "," + str(cores) + "\n")
        except subprocess.CalledProcessError as e:
            print(f"Error in updating! Error is {e.output} {e.cmd} {e.returncode}")

    return


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


def schedule(is_done):
    timer = threading.Timer(T_SAMPLE_SECONDS, schedule, (is_done,))
    timer.start()
    progress = read_progress()
    updated_progress_time = time.time() - start_time

    if progress != TOTAL_PROGRESS:
        update(progress)
    else:
        timer.cancel()
        alloc_time = time.time() - start_time
        with open(ALLOCATIONS_FILE, "a+") as file:
            file.write(str(alloc_time) + "," + str(cores) + "\n")
            file.write(str(alloc_time) + "," + str(0) + "\n")
        is_done.value = 1
        return
    with open(PROGRESS_TIMELINE, "a+") as file:
        file.write(str(updated_progress_time) + "," + str(progress) + "\n")


def start(model: str, num_batches: int, batch_size: int, desired_deadline: float, alpha: float,
          epochs: int, allocations_timeline: str, progress_timeline: str, image_name: str, dl_change=False,
          progress_file=None):
    global ALLOCATIONS_FILE
    global PROGRESS_FILE
    global PROGRESS_TIMELINE
    global start_time
    global TOTAL_PROGRESS
    global deadline
    global csi_old
    global cores
    global time_units

    time_units = 0
    TOTAL_PROGRESS = epochs * num_batches
    deadline = desired_deadline * alpha
    csi_old = MAX_CORES
    cores = MAX_CORES
    is_done = Value('i', 0)
    ALLOCATIONS_FILE = allocations_timeline
    PROGRESS_TIMELINE = progress_timeline
    if progress_file is not None:
        filename = progress_file
    else:
        filename = "progress.txt"
    PROGRESS_FILE = os.path.join(results_dir, filename)
    try:
        print(f"Running Docker.")
        subprocess.run('docker run -d -v $(pwd)/results:/project/results --name='
                       + container_name + ' ' + image_name + ' ' + model + ' '
                       + str(num_batches) + ' ' + str(epochs) + ' ' + str(batch_size) + ' ' + filename,
                       shell=True, check=True, capture_output=True)

    except subprocess.CalledProcessError:
        print(f"Error during docker opening \n")
        subprocess.run(' docker rm -vf $(docker ps -aq)', shell=True)
        return

    with open(ALLOCATIONS_FILE, "a+") as file:
        file.write("0," + str(MAX_CORES) + "\n")
    with open(PROGRESS_TIMELINE, "a+") as file:
        file.write("0,0\n")
    # Training starts when multi_gpu creates a file with progress 0.
    read_progress()
    print(f"Starting Deadline Control at deadline {desired_deadline}s with alpha = {alpha}.")
    start_time = time.time()
    schedule(is_done)
    time.sleep(T_SAMPLE_SECONDS)

    # This section can be removed after the dynamic change is moved outside this module.
    # This is due to the fact that scheduling is already done by the schedule() function.
    while not bool(is_done.value):
        time.sleep(T_SAMPLE_SECONDS)
        # Dynamically change deadline. After prototyping this should be moved outside controller.
        if time_units / deadline >= 0.3 and dl_change:
            deadline = desired_deadline * 0.8
            print(f"deadline changed to {deadline}\n")
            break
    if dl_change:
        while not bool(is_done.value):
            time.sleep(T_SAMPLE_SECONDS)
    end = time.time() - start_time

    os.remove(PROGRESS_FILE)
    subprocess.run(' docker rm -f ' + container_name, shell=True, stdout=subprocess.DEVNULL)
    print(f"Finished Training at time {end}")
    return end