#! /usr/bin/env python3

import cmd
import argparse
import shlex
import os
import subprocess
import multiprocessing as mp
import threading
import time
import uuid
from typing import List
import logging
from datetime import datetime
from controller import *
from job import TrainingJob


logger = logging.getLogger(__name__)


class Shell(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = ">>> "
        self.intro = "Welcome to DECORN-NN shell. Type 'help' to see the commands."
        
        base_results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cli_results")
        #timestamp format: 2025-09-05_14-30-45
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_path = os.path.join(base_results_path, timestamp)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.jobs: List[TrainingJob] = []
        self.jobs_lock = threading.Lock()
        self._build_docker_image()
        
        # Start scheduler
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=schedule, args=(self,), daemon=True)
        self.scheduler_thread.start()

    def _build_docker_image(self):
        print("Building Docker image...")
        try:
            subprocess.run(
                f"docker build -t {IMAGE} "
                f"--cpu-quota={int(100000 * mp.cpu_count())} "
                f"--build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . ",
                shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("Docker image built successfully.")
        except subprocess.CalledProcessError as e:
            print(f"FATAL: Docker build failed. Error: {e}")
            exit(1)

    def do_start_training(self, line):
        """
        Start a training run. Can be called multiple times for concurrent jobs.
        Usage: start_training -m resnet50 -nb 10 -bs 32 -dd 20.0 -e 2
        Usage: start_training -m resnet50 -nb 2 -bs 2 -dd 1.5 -e 2
        Usage: start_training -m resnet50 -nb 20 -bs 32 -dd 583.79 -e 5
        Usage: start_training -m resnet50 -nb 10 -bs 35 -dd 119.35 -e 2
        """
        parser = argparse.ArgumentParser(prog="start_training", add_help=False)
        parser.add_argument("-m", "--model", required=True)
        parser.add_argument("-nb", "--num_batches", type=int, required=True)
        parser.add_argument("-bs", "--batch_size", type=int, required=True)
        parser.add_argument("-dd", "--desired_deadline", type=float, required=True)
        parser.add_argument("-a", "--alpha", type=float, default=1.0)
        parser.add_argument("-e", "--epochs", type=int, required=True)
        parser.add_argument("-d", "--dl_change", action="store_true")

        try:
            args = parser.parse_args(shlex.split(line))
        except SystemExit:
            print("Invalid arguments. Type 'help start_training' for usage.")
            return
        with self.jobs_lock:
            job_id = str(uuid.uuid4().hex[:8])
            run_path = os.path.join(self.results_path, f"run_{len(self.jobs)}_{job_id}")
            
            job = TrainingJob(job_id, run_path, args)
            
            try:
                job.launch()
                self.jobs.append(job)
                print(f"Successfully launched job {job.id} with container {job.container_name}.")
            except Exception as e:
                print(f"Failed to launch job {job.id}. Error: {e}")
                job.stop()

    def do_status(self, line):
        """Shows the status of all active training jobs."""
        with self.jobs_lock:
            if not self.jobs:
                print("No active training jobs.")
                return
            
            print(f"{'JOB ID':<10} {'CONTAINER':<25} {'PROGRESS':<15} {'CORES':<10}")
            print("-" * 60)
            for job in self.jobs:
                progress = 0 if read_progress(job) == -1 else read_progress(job)
                progress_percent = (progress / job.total_progress * 100)
                progress_str = f"{progress}/{job.total_progress} ({progress_percent:.1f}%)"
                print(f"{job.id:<10} {job.container_name:<25} {progress_str:<15} {job.current_cores:<10.2f}")

    def do_exit(self, arg):
        """Stops all running jobs and exits the shell."""
        self.scheduler_active = False
        with self.jobs_lock:
            for job in self.jobs:
                job.is_done = True
                job.stop()
        # Wait up to 2 seconds for the scheduler thread to finish (without forcing it to stop)
        self.scheduler_thread.join(timeout=2)
        return True
    
    def emptyline(self):
        pass

if __name__ == "__main__":
    try:
        Shell().cmdloop()
    except KeyboardInterrupt:
        print("\t")
        subprocess.run(
            'docker ps -a -q --filter "name=decorn-nn-job-*" | xargs -r docker rm -vf',
            shell=True,
            stdout=subprocess.DEVNULL
        )