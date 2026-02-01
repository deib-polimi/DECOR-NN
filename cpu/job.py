from controller import *

class TrainingJob:
    """A class to encapsulate the state of a single training run."""
    def __init__(self, job_id, run_path, args):
        self.id = job_id
        self.args = args
        self.container_name = f"decorn-nn-job-{self.id}"
        
        # Paths
        self.run_path = run_path
        self.allocations_file = os.path.join(run_path, "allocations_timeline.csv")
        self.progress_timeline_file = os.path.join(run_path, "progress_timeline.csv")
        self.progress_file = os.path.join(run_path, "progress.txt")

        self.total_progress = args.epochs * args.num_batches
        self.desired_deadline = args.desired_deadline 
        self.deadline = self.desired_deadline * args.alpha
        self.start_time = None
        self.time_units = 0
        self.dynamic_dl = args.dl_change
        self.dl_changed = False
        self.is_done = False
        self.current_cores = MAX_CORES
        self.csi_old = MAX_CORES
        self.csp = 0
        self.tot_time = 0

    def launch(self):
        """Launches the Docker container for this job."""
        os.makedirs(self.run_path, exist_ok=True)
        # Write initial allocation
        with open(self.allocations_file, "w") as f:
            f.write(str(time.monotonic()) + "\n")
            f.write("time,cores\n0," + str(self.current_cores) + "\n")
            f.write(f"{self.args.model},{self.args.num_batches},{self.args.epochs},{self.args.batch_size},{self.args.desired_deadline}\n")
        with open(self.progress_timeline_file, "w") as f:
            f.write("time,progress\n0,0\n")
            f.write(f"{self.args.model},{self.args.num_batches},{self.args.epochs},{self.args.batch_size},{self.args.desired_deadline}\n")

        start(
            model=self.args.model,
            num_batches=self.args.num_batches,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            container_name=self.container_name,
            progress_file_path=self.progress_file,
            image_name=IMAGE           
        )

    def read_progress(self):
        """Reads the progress from the job's progress file."""
        try:
            with open(self.progress_file, "r") as file:
                return int(file.readline())
        except (ValueError, FileNotFoundError):
            return -1 # Indicates not started or file not ready

    def stop(self):
        """Stops and removes the container and cleans up files."""
        #print(f"[{self.container_name}] Stopping and cleaning up...")
        subprocess.run(f'docker rm -f {self.container_name}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

