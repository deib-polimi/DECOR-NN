import subprocess
import time
from controller import MAX_CORES, T_SAMPLE_SECONDS, CPU_PERIOD, MIN_CORES, MAX_CORES

LAST_CORE_ALLOCATION = 0

def schedule_edf(shell):
    global LAST_CORE_ALLOCATION
    LAST_CORE_ALLOCATION = 0
    while shell.scheduler_active:
        time.sleep(T_SAMPLE_SECONDS)
        
        with shell.jobs_lock:
            # Rimuovere job terminati
            for job in shell.jobs:
                progress = job.read_progress()
                if progress >= job.total_progress:
                    job.is_done = True
                    end_time = time.monotonic()
                    job.tot_time = end_time - job.start_time
                    print(f"[{job.container_name}] Finished Training in {job.tot_time:.2f}s at {end_time:.2f}s")
                    
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{job.tot_time},{job.current_cores}\n")
                        f.write(f"{job.tot_time},0\n")
                    
                    job.stop()
                else:
                    if progress == -1 and job.start_time is None:
                        continue
                    if job.start_time is None: # First time
                        job.start_time = time.monotonic()
            
            shell.jobs = [job for job in shell.jobs if not job.is_done]
            
            if not shell.jobs:
                continue
            
            # Calcolare numero di job attivi
            active_jobs = [job for job in shell.jobs if job.start_time is not None]
            num_active_jobs = len(active_jobs)
            
            if num_active_jobs == 0:
                continue
            
            # Se c'è un solo job, dargli tutti i core
            if num_active_jobs == 1:
                cores_per_job = MAX_CORES
                cpu_quota = int(cores_per_job * CPU_PERIOD)
                
                if cores_per_job != LAST_CORE_ALLOCATION:
                    LAST_CORE_ALLOCATION = cores_per_job
                    now = time.monotonic()
                    job = active_jobs[0]
                    
                    print(f"Current allocation: [{job.container_name}] = {cores_per_job}")
                    subprocess.run(
                        f'docker update --cpu-quota="{cpu_quota}" {job.container_name}',
                        shell=True, check=True, capture_output=True
                    )
                    
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{now-job.start_time},{job.current_cores}\n")
                        f.write(f"{now-job.start_time},{cores_per_job}\n")
                    
                    job.current_cores = cores_per_job
                continue
            
            # Trovare il job con la deadline più vicina (in tempo assoluto)
            # absolute_deadline = start_time + desired_deadline
            earliest_job = min(
                active_jobs,
                key=lambda j: j.start_time + j.desired_deadline
            )
            
            # Assegnare MIN_CORES a tutti gli altri job
            cores_given_to_others = MIN_CORES * (num_active_jobs - 1)
            
            # Assegnare i core rimanenti al job con la deadline più vicina
            cores_for_earliest = MAX_CORES - cores_given_to_others
            
            # Job con deadline più stringente ha almeno MIN_CORES
            cores_for_earliest = max(cores_for_earliest, MIN_CORES)
            
            # Verificare se l'allocazione è cambiata
            allocation_changed = False
            now = time.monotonic()
            
            for job in active_jobs:
                if job == earliest_job:
                    new_cores = cores_for_earliest
                else:
                    new_cores = MIN_CORES
                
                new_cores = (new_cores * 100) // 1 / 100  # Arrotondare a 2 decimali
                cpu_quota = int(new_cores * CPU_PERIOD)
                
                if job.current_cores != new_cores:
                    allocation_changed = True
                    
                    print(f"[{job.container_name}] allocation: {new_cores} (deadline: {job.start_time + job.desired_deadline:.2f}s)")
                    
                    subprocess.run(
                        f'docker update --cpu-quota="{cpu_quota}" {job.container_name}',
                        shell=True, check=True, capture_output=True
                    )
                    
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{now-job.start_time},{job.current_cores}\n")
                        f.write(f"{now-job.start_time},{new_cores}\n")
                    
                    job.current_cores = new_cores
            
            if allocation_changed:
                LAST_CORE_ALLOCATION = cores_for_earliest

def schedule_proportional(shell):
    global LAST_CORE_ALLOCATION
    LAST_CORE_ALLOCATION = 0
    while shell.scheduler_active:
        time.sleep(T_SAMPLE_SECONDS)
        
        with shell.jobs_lock:
            # Rimuovere job terminati
            for job in shell.jobs:
                progress = job.read_progress()
                if progress >= job.total_progress:
                    job.is_done = True
                    end_time = time.monotonic() 
                    job.tot_time = end_time - job.start_time
                    print(f"[{job.container_name}] Finished Training in {job.tot_time:.2f}s at {end_time:.2f}s")
                    with open(job.allocations_file, "a") as f:
                        f.write(f"{job.tot_time},{job.current_cores}\n")
                        f.write(f"{job.tot_time},0\n")
                    job.stop()
                else:
                    if progress == -1 and job.start_time is None:
                        continue
                    if job.start_time is None: # First time
                        job.start_time = time.monotonic()

            shell.jobs = [job for job in shell.jobs if not job.is_done]

            if not shell.jobs:
                continue
            
            # Calcolare numero di job attivi
            num_active_jobs = len([job for job in shell.jobs if job.start_time is not None])
            if num_active_jobs == 0:
                continue
            
            # Assegnare core in modo proporzionale
            cores_per_job = ( MAX_CORES / num_active_jobs * 100) // 1 / 100
            cpu_quota = int(cores_per_job * CPU_PERIOD)

            if cores_per_job != LAST_CORE_ALLOCATION:
                LAST_CORE_ALLOCATION = cores_per_job
                now = time.monotonic() 
                for job in shell.jobs:
                    if job.start_time is not None:
                        print(f"Current allocation: {LAST_CORE_ALLOCATION}")
                        subprocess.run(f'docker update --cpu-quota="{cpu_quota}" {job.container_name}',
                        shell=True, check=True, capture_output=True)
                        with open(job.allocations_file, "a") as f:
                            f.write(f"{now-job.start_time},{job.current_cores}\n")
                            f.write(f"{now-job.start_time},{cores_per_job}\n")
                        job.current_cores = cores_per_job
