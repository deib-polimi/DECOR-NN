"""
To use this script, comment the following line in cpu/shell.py:
        # self.scheduler_thread = threading.Thread(target=schedule, args=(self,), daemon=True)
        # self.scheduler_thread.start()
"""
import os
import threading
import time
import uuid
from shell import Shell, TrainingJob
from controller import T_SAMPLE_SECONDS, read_progress, schedule
import random
from scheduler import *

DIRECTORY_NAME = ""
SEED = 4
random.seed(SEED)

def job_laucher_generator():
    return [random.randint(0, 3) for _ in range(14)]
        

def launch_jobs():

    # Define the training jobs to launch
    job_configs = [
        {
            "model": "resnet50",
            "num_batches": 100,
            "batch_size": 32,
            "desired_deadline": 0.1,
            "alpha": 1.0,
            "epochs": 2,
            "dl_change": False
        },
        {
            "model": "vgg19",
            "num_batches": 100,
            "batch_size": 32,
            "desired_deadline": 0.1,
            "alpha": 1.0,
            "epochs": 2,
            "dl_change": False
        },
        {
            "model": "inception_v3",
            "num_batches": 100,
            "batch_size": 32,
            "desired_deadline": 0.1,
            "alpha": 1.0,
            "epochs": 2,
            "dl_change": False
        }        
    ]

    launcher_sequence = job_laucher_generator()

    BASELINE_MEAN = 0
    shell = Shell(DIRECTORY_NAME + "/baseline")
    # 1. ciclo for per calcolare le baseline 
    for i, config in enumerate(job_configs):
        shell.scheduler_thread = threading.Thread(target=schedule, args=(shell,), daemon=True)
        shell.scheduler_thread.start()
        try:
            job_id = f"{config['model']}_{str(uuid.uuid4().hex[:8])}"
            run_path = os.path.join(shell.results_path, f"{job_id}")
            args = type("Args", (object,), config)

            job = TrainingJob(job_id, run_path, args)
            with shell.jobs_lock:
                job.launch()
                shell.jobs.append(job)
                print(f"Successfully launched job {job.id} with container {job.container_name}.")
                job.start_time = time.time()
                while(job.read_progress() < job.total_progress):
                    time.sleep(3)
                progress = job.read_progress()
                job.is_done = True
                end_time = time.time() 
                tot_time = end_time - job.start_time
                print(f"[{job.container_name}] Finished Training in {tot_time:.2f}s at {end_time:.2f}s")
                with open(job.allocations_file, "a") as f:
                    f.write(f"{tot_time},{job.current_cores}\n")
                    f.write(f"{tot_time},0\n")
                job.stop()

                elapsed_time = time.time() - job.start_time
                with open(job.progress_timeline_file, "a") as f:
                    f.write(f"{elapsed_time},{progress}\n")
                
                # Aggiornare deadline di job_configs con le nuove baseline calcolate
                job_configs[i]['desired_deadline'] = tot_time * 2
                BASELINE_MEAN += tot_time
        except Exception as e:
            print(f"Failed to launch job {job_id}. Error: {e}")    
    shell.scheduler_thread.join(timeout=2)
    
    BASELINE_MEAN /= 3
    
    
    # 2. ciclo for per lanciare job con schedule ()
    shell = Shell(DIRECTORY_NAME + "/scheduled")
    shell.scheduler_thread = threading.Thread(target=schedule, args=(shell,), daemon=True)
    shell.scheduler_thread.start()
    for i, config_index in enumerate(launcher_sequence):
        if(config_index < 3):
            try:
                job_id = f"scheduled_job_{i}_{job_configs[config_index]['model']}_{str(uuid.uuid4().hex[:8])}"
                run_path = os.path.join(shell.results_path, f"run_scheduled_{job_id}")
                args = type("Args", (object,), job_configs[config_index])

                job = TrainingJob(job_id, run_path, args)
                with shell.jobs_lock:
                    job.launch()
                    shell.jobs.append(job)
                    print(f"Successfully launched job {job.id} with container {job.container_name}.")
            except Exception as e:
                print(f"Failed to launch job {job_id}. Error: {e}")
        time.sleep(BASELINE_MEAN/4)
    shell.scheduler_thread.join(timeout=2)

    # 3. ciclo for per lanciare job con schedule_proportional ()
    # 4. ciclo for per lanciare job con schedule_edf ()
 
    print("Done!\n")

if __name__ == "__main__":
    DIRECTORY_NAME = "esperimento"
    launch_jobs()