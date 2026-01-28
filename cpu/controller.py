#! /usr/bin/env python3

import os
import subprocess
import math
import multiprocessing as mp
import time
import threading

# --- Constants ---
T_SAMPLE_SECONDS = 1
MIN_CORES = 1
MACHINE_CORES = float(mp.cpu_count())
K = 50
TI_SECONDS = 12
CORE_QUANTUM = 0.05
QUANTUM_DIGITS = -int(math.floor(math.log10(CORE_QUANTUM)))
SPIKE_PERCENTAGE = 0.99
CPU_PERIOD = 100000
IMAGE = 'test'
STARTING_CORE = 2
ENDING_CORE = MACHINE_CORES - 1
MAX_CORES = ENDING_CORE - STARTING_CORE + 1

def next_allocation(progress, total_progress, set_point, job):
    """
    Calculates the next desired CPU core allocation for a job.
    """
    csp = K * (set_point - (progress / total_progress))
    csi = job.csi_old + csp * (T_SAMPLE_SECONDS / TI_SECONDS)
    cs = min(max(MIN_CORES, csp + csi), MAX_CORES)

    #cs = round(round(cs / CORE_QUANTUM) * CORE_QUANTUM, QUANTUM_DIGITS)
    # We round later to increase precision and avoid rounding twice
    cs = (cs / CORE_QUANTUM) * CORE_QUANTUM
   
    if set_point >= 1.0:
        cs = job.current_cores
        if progress / total_progress < SPIKE_PERCENTAGE:
            return MAX_CORES, csp

    return cs, csp


def update(desired, scaling_factor, job):
    actual_cores = desired * scaling_factor
    # Round the final allocation
    job.csi_old = actual_cores - job.csp
    
    quantized_cores = round(actual_cores , QUANTUM_DIGITS)
    final_cores = max(MIN_CORES, quantized_cores) #critical when having N jobs where N is higher than the number of cores

    cpu_quota = int(final_cores * CPU_PERIOD)

    subprocess.run(f'docker update --cpu-quota="{cpu_quota}" {job.container_name}',
                    shell=True, check=True, capture_output=True)

    if final_cores != job.current_cores:
        try:
            
            alloc_time = time.time() - job.start_time
            with open(job.allocations_file, "a") as f:
                f.write(f"{alloc_time},{job.current_cores}\n") #is it needed? In the .csv the prev file tells the amount of cores allocated
                f.write(f"{alloc_time},{final_cores}\n")
            
            job.current_cores = final_cores
        except subprocess.CalledProcessError as e:
            print(f"[{job.container_name}] Error updating CPU quota: {e.stderr.decode()}")


def read_progress(job):
    """Reads the progress from the job's progress file."""
    try:
        with open(job.progress_file, "r") as file:
            return int(file.readline())
    except (ValueError, FileNotFoundError):
        return -1 # Indicates not started or file not ready


def schedule(self):
    """The heart of the controller. Manages CPU for all jobs."""
    while self.scheduler_active:
        time.sleep(T_SAMPLE_SECONDS)
        
        with self.jobs_lock:
            active_jobs = [job for job in self.jobs if not job.is_done]
            if not active_jobs:
                continue

            # 1. Calculate desired cores for each job
            desired_allocations = {}
            total_desired_cores = 0
            
            for job in active_jobs:
                progress = read_progress(job)

                if progress == -1 and job.start_time is None:
                    continue # Job hasn't created its progress file yet
                
                if job.start_time is None: # First time
                    job.start_time = time.time()

                # Check for completion
                if progress >= job.total_progress:
                    job.is_done = True
                    end_time = time.time() 
                    tot_time = end_time - job.start_time
                    print(f"[{job.container_name}] Finished Training in {tot_time:.2f}s at {end_time:.2f}s")
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{tot_time},{job.current_cores}\n")
                        f.write(f"{tot_time},0\n")
                    job.stop()
                    continue

                # Update job timeline
                elapsed_time = time.time() - job.start_time
                with open(job.progress_timeline_file, "a") as f:
                    f.write(f"{elapsed_time},{progress}\n")

                #Dynamically change deadline
                if job.time_units / job.deadline >= 0.3 and job.dynamic_dl and not job.dl_changed:
                    job.deadline = job.desired_deadline * 0.8
                    print(f"[{job.container_name}] Deadline changed to {job.deadline}\n")
                    job.dl_changed = True

                #Desired allocation
                job.time_units = time.time() - job.start_time 
                set_point = job.time_units / job.deadline
                desired_cores, csp = next_allocation(progress, job.total_progress, set_point, job)
                job.csp = csp

                desired_allocations[job.id] = desired_cores
                total_desired_cores += desired_cores

            # 2. Calculate proportional allocation
            scaling_factor = 1.0
            if total_desired_cores > MAX_CORES:
                scaling_factor = MAX_CORES / total_desired_cores
            
            # 3. Apply the new allocations
            for job in active_jobs:
                if job.id not in desired_allocations:
                    continue
                else:
                    update(desired_allocations[job.id], scaling_factor, job)

            # Clean up finished jobs from the main list
            self.jobs = [job for job in self.jobs if not job.is_done]


def start(model: str, num_batches: int, batch_size: int, epochs: int, 
          container_name: str, progress_file_path: str, image_name: str):
    """
    Launches the Docker container for a single training job.
    It no longer manages the lifecycle or scheduling.
    """
    progress_dir = os.path.dirname(progress_file_path)
    progress_filename = os.path.basename(progress_file_path)

    try:
        docker_command = (
            f'docker run -d -v {progress_dir}:/project/results --name={container_name} '
            f'--cpu-period=100000 --cpu-quota={int(MAX_CORES * 100000)} '
            f'--cpuset-cpus="{int(STARTING_CORE)}-{int(ENDING_CORE)}" '
            f'{image_name} {model} {str(num_batches)} {str(epochs)} {str(batch_size)} {progress_filename}'
        )
        subprocess.run(docker_command, shell=True, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"[{container_name}] Error during Docker container startup: {e.stderr.decode()}")
        raise

    return container_name
