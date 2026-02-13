#! /usr/bin/env python3

import os
import subprocess
import math
import multiprocessing as mp
import time
import threading
import math

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

def round_with_constraint(desired_allocations, scaling_factor):
    # Extract keys and values
    keys = list(desired_allocations.keys())
    values = list(desired_allocations.values())
    
    # Apply the scaling factor
    scaled_values = [x * scaling_factor for x in values]
    
    n = len(scaled_values)
    
    # Calculate ceiling rounding and the "loss" if we round down
    rounded_up = [math.ceil(x) for x in scaled_values]
    losses = [x - math.floor(x) for x in scaled_values]  # Loss when rounding down
    
    # Sum with all values rounded up
    sum_rounded_up = sum(rounded_up)
    
    # If the sum already satisfies the constraint, return everything rounded up
    if sum_rounded_up <= MAX_CORES:
        return {keys[i]: rounded_up[i] for i in range(n)}
    
    # Otherwise, we need to round down some values
    # Choose those with minimum loss (e.g., 5.1 instead of 5.9)
    excess = sum_rounded_up - MAX_CORES
    
    # Create list of indices sorted by increasing loss
    sorted_indices = sorted(range(n), key=lambda i: losses[i])
    
    result = rounded_up.copy()
    
    # Round down the values with minimum loss
    for i in sorted_indices:
        if excess <= 0:
            break
        # Round down this value
        result[i] = math.floor(scaled_values[i])
        excess -= 1  # Each change from ceil to floor reduces the sum by 1
    
    # Return as dictionary
    return {keys[i]: result[i] for i in range(n)}

def next_allocation(progress, total_progress, set_point, job):
    """
    Calculates the next desired CPU core allocation for a job.
    """
    csp = K * (set_point - (progress / total_progress))
    csi = job.csi_old + csp * (T_SAMPLE_SECONDS / TI_SECONDS)
    cs = min(max(MIN_CORES, csp + csi), MAX_CORES)

    #cs = round(round(cs / CORE_QUANTUM) * CORE_QUANTUM, QUANTUM_DIGITS)
    # We round later to increase precision and avoid rounding twice
    #cs = (cs / CORE_QUANTUM) * CORE_QUANTUM
   
    if set_point >= 1.0:
        cs = job.current_cores
        if progress / total_progress < SPIKE_PERCENTAGE:
            return MAX_CORES, csp

    return cs, csp


def update(desired, job, last_used_core):
    """actual_cores = desired * scaling_factor
    # Round the final allocation
    job.csi_old = actual_cores - job.csp
    
    quantized_cores = round(actual_cores , QUANTUM_DIGITS)
    final_cores = max(MIN_CORES, quantized_cores) #critical when having N jobs where N is higher than the number of cores

    cpu_quota = int(final_cores * CPU_PERIOD)"""
    if desired != job.current_cores:
        try: 
            subprocess.run(f'docker update --cpuset-cpus="{last_used_core}-{last_used_core + desired - 1}" {job.container_name}',
                shell=True, check=True, capture_output=True)
            print(f"[{job.container_name}] {desired:.2f} cores")
            alloc_time = time.monotonic() - job.start_time
            with open(job.allocations_file, "a") as f:
                f.write(f"{alloc_time},{job.current_cores}\n")
                f.write(f"{alloc_time},{desired}\n")
            
            job.current_cores = desired
            return last_used_core + desired
        except subprocess.CalledProcessError as e:
            print(f"[{job.container_name}] Error updating CPU quota: {e.stderr.decode()}")


def read_progress(job):
    """Reads the progress from the job's progress file."""
    try:
        with open(job.progress_file, "r") as file:
            return int(file.readline())
    except (ValueError, FileNotFoundError):
        return -1 # Indicates not started or file not ready


def schedule(shell):
    """The heart of the controller. Manages CPU for all jobs."""
    while shell.scheduler_active:
        time.sleep(T_SAMPLE_SECONDS)
        
        with shell.jobs_lock:
            shell.jobs = [job for job in shell.jobs if not job.is_done]
            if not shell.jobs:
                continue

            # 1. Calculate desired cores for each job
            desired_allocations = {}
            total_desired_cores = 0
            
            for job in shell.jobs:
                progress = read_progress(job)

                if progress == -1 and job.start_time is None:
                    continue # Job hasn't created its progress file yet
                
                if job.start_time is None: # First time
                    job.start_time = time.monotonic()

                # Check for completion
                if progress >= job.total_progress:
                    job.is_done = True
                    end_time = time.monotonic() 
                    job.tot_time = end_time - job.start_time
                    print(f"[{job.container_name}] Finished Training in {job.tot_time:.2f}s at {end_time:.2f}s")
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{job.tot_time},{job.current_cores}\n")
                        f.write(f"{job.tot_time},0\n")
                    job.stop()
                    continue

                # Update job timeline
                elapsed_time = time.monotonic() - job.start_time
                with open(job.progress_timeline_file, "a") as f:
                    f.write(f"{elapsed_time},{progress}\n")

                #Dynamically change deadline
                if job.time_units / job.deadline >= 0.3 and job.dynamic_dl and not job.dl_changed:
                    job.deadline = job.desired_deadline * 0.8
                    print(f"[{job.container_name}] Deadline changed to {job.deadline}\n")
                    job.dl_changed = True

                #Desired allocation
                job.time_units = time.monotonic() - job.start_time 
                set_point = job.time_units / job.deadline
                desired_cores, csp = next_allocation(progress, job.total_progress, set_point, job)
                job.csp = csp

                desired_allocations[job.id] = desired_cores
                total_desired_cores += desired_cores

            # 2. Calculate proportional allocation
            scaling_factor = 1.0
            if total_desired_cores > MAX_CORES:
                scaling_factor = MAX_CORES / total_desired_cores

            desired_allocations = round_with_constraint(desired_allocations, scaling_factor)
            
            last_used_core = STARTING_CORE
            # 3. Apply the new allocations
            for job in shell.jobs:
                if job.id not in desired_allocations:
                    continue
                else:
                    last_used_core = update(desired_allocations[job.id], job, last_used_core)



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
            f'{image_name} {model} {str(num_batches)} {str(epochs)} {str(batch_size)} {progress_filename}'
        )
        subprocess.run(docker_command, shell=True, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"[{container_name}] Error during Docker container startup: {e.stderr.decode()}")
        raise

    return container_name
