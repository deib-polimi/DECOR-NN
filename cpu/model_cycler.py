"""
To use this script, comment the following line in cpu/shell.py:
        # self.scheduler_thread = threading.Thread(target=schedule, args=(self,), daemon=True)
        # self.scheduler_thread.start()
"""
import os
import threading
import time
import traceback
import uuid
from shell import Shell, TrainingJob
from controller import T_SAMPLE_SECONDS, read_progress, schedule
import random
from scheduler import *

BASELINE_MEAN = 0.0
DIRECTORY_NAME = ""
SEED = 4
random.seed(SEED)

TICK_FACTOR = 0.4
DEADLINE_FACTOR = 2

# Define the training jobs to launch
job_configs = [
    {
        "model": "resnet50",
        "num_batches": 10,
        "batch_size": 10,
        "desired_deadline": 0.1,
        "alpha": 1.0,
        "epochs": 2,
        "dl_change": False
    },
    {
        "model": "resnet50",
        "num_batches": 10,
        "batch_size": 10,
        "desired_deadline": 0.1,
        "alpha": 1.0,
        "epochs": 2,
        "dl_change": False
    },
    {
        "model": "resnet50",
        "num_batches": 10,
        "batch_size": 10,
        "desired_deadline": 0.1,
        "alpha": 1.0,
        "epochs": 2,
        "dl_change": False
    }
]

def job_laucher_generator(range_val):
    #return [random.randint(0, len(job_configs)) for _ in range(range_val)]
    return [0]    

def launcher(shell, launcher_sequence, name):
    global BASELINE_MEAN
    for i, config_index in enumerate(launcher_sequence):
        if(config_index < len(job_configs)):
            try:
                job_id = f"{name}_job_{i}_{job_configs[config_index]['model']}_{str(uuid.uuid4().hex[:8])}"
                run_path = os.path.join(shell.results_path, f"run_{name}_{job_id}")
                args = type("Args", (object,), job_configs[config_index])

                job = TrainingJob(job_id, run_path, args)
                with shell.jobs_lock:
                    job.launch()
                    shell.jobs.append(job)
                    print(f"Successfully launched job {job.id} with container {job.container_name}.")
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
        time.sleep(BASELINE_MEAN * TICK_FACTOR)

def launch_jobs(launcher_sequence):
    global BASELINE_MEAN

    # 1. ciclo for per calcolare le baseline 
    """shell = Shell(DIRECTORY_NAME + "/baseline")
    for i, config in enumerate(job_configs):
        shell.scheduler_thread = threading.Thread(target=schedule, args=(shell,), daemon=True)
        shell.scheduler_thread.start()
        try:
            job_id = f"{config['model']}_{str(uuid.uuid4().hex[:8])}"
            run_path = os.path.join(shell.results_path, f"{job_id}")
            args = type("Args", (object,), config)

            job = TrainingJob(job_id, run_path, args)
            job.launch()
            shell.jobs.append(job)
            job.start_time = time.monotonic()
            print(f"Successfully launched job {job.id} with container {job.container_name} at {job.start_time}.")
            while(job.read_progress() < job.total_progress):
                time.sleep(3)
                
            # Aggiornare deadline di job_configs con le nuove baseline calcolate
            job_configs[i]['desired_deadline'] = job.tot_time * DEADLINE_FACTOR
            BASELINE_MEAN += job.tot_time
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
    shell.scheduler_thread.join(timeout=2)
    

    BASELINE_MEAN /= len(job_configs)
    """
    BASELINE_MEAN = 44.0
    for i, config_index in enumerate(launcher_sequence):
        job_configs[config_index]['desired_deadline'] = BASELINE_MEAN * DEADLINE_FACTOR
    
    
    # 2. ciclo for per lanciare job con schedule ()
    shell = Shell(DIRECTORY_NAME + "/scheduled")
    shell.scheduler_thread = threading.Thread(target=schedule, args=(shell,), daemon=True)
    shell.scheduler_thread.start()

    launcher(shell, launcher_sequence, "scheduled")

    while True:
        if not shell.jobs:
            break
        time.sleep(10)
    shell.scheduler_thread.join(timeout=2)

    # 3. ciclo for per lanciare job con schedule_proportional ()
    shell = Shell(DIRECTORY_NAME + "/proportional")
    shell.scheduler_thread = threading.Thread(target=schedule_proportional, args=(shell,), daemon=True)
    shell.scheduler_thread.start()

    launcher(shell, launcher_sequence, "proportional")

    while True:
        if not shell.jobs:
            break
        time.sleep(10)
    shell.scheduler_thread.join(timeout=2)

    # 4. ciclo for per lanciare job con schedule_edf ()
    shell = Shell(DIRECTORY_NAME + "/edf")
    shell.scheduler_thread = threading.Thread(target=schedule_edf, args=(shell,), daemon=True)
    shell.scheduler_thread.start()

    launcher(shell, launcher_sequence, "edf")

    while True:
        if not shell.jobs:
            break
        time.sleep(10)
    shell.scheduler_thread.join(timeout=2)
 
    print("Done!\n")

if __name__ == "__main__":
    DIRECTORY_NAME = "esperimento"
    launcher_sequence = job_laucher_generator(14)
    launch_jobs(launcher_sequence)