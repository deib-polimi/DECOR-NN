import subprocess
import time
from controller import MAX_CORES, T_SAMPLE_SECONDS, CPU_PERIOD

LAST_CORE_ALLOCATION = 0

def schedule_edf():
    pass

def schedule_proportional(shell):
    global LAST_CORE_ALLOCATION
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
