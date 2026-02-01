import subprocess
import time
from controller import MAX_CORES, T_SAMPLE_SECONDS, CPU_PERIOD

LAST_CORE_ALLOCATION = 0

def schedule_edf():
    pass

def schedule_proportional(shell):
    while shell.scheduler_active:
        time.sleep(T_SAMPLE_SECONDS)
        
        with shell.jobs_lock:
            shell.jobs = [job for job in shell.jobs if not job.is_done]

            if not shell.jobs:
                continue
            
            # Calcolare numero di job attivi
            num_active_jobs = len(shell.jobs)
            if num_active_jobs == 0:
                continue
            
            # Assegnare core in modo proporzionale
            cores_per_job = ( MAX_CORES / num_active_jobs * 100) // 1 / 100
            cpu_quota = int(cores_per_job * CPU_PERIOD)

            if cores_per_job != LAST_CORE_ALLOCATION:
                LAST_CORE_ALLOCATION = cores_per_job
                for job in shell.jobs:
                    print(f"Current allocation: {LAST_CORE_ALLOCATION}")
                    subprocess.run(f'docker update --cpu-quota="{cpu_quota}" {job.container_name}',
                    shell=True, check=True, capture_output=True)
