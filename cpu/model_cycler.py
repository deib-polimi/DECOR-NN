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
BASELINE_MEAN = 40

launcher_sequence = [0]

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
    # Aggiornare deadline di job_configs con le nuove baseline calcolate
    for config in job_configs:
        config["desired_deadline"] = BASELINE_MEAN * DEADLINE_FACTOR
    
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
 
    print("Done!\n")
    subprocess.run(f"zip -r {DIRECTORY_NAME}.zip {DIRECTORY_NAME}", shell=True, check=True, capture_output=True)
    print(f"Results zipped into {DIRECTORY_NAME}.zip")

if __name__ == "__main__":
    DIRECTORY_NAME = "esperimento"
    launch_jobs(launcher_sequence)