import subprocess
import multiprocessing as mp
import os
from datetime import datetime
import uuid
import sys

IMAGE = 'deadline'
MAX_CORES = 1

job_id = str(uuid.uuid4().hex[:8])
print(job_id)
base_results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_deadline")
run_path = os.path.join(base_results_path, f"run_{job_id}")
progress_file = os.path.join(run_path, "progress.txt")

progress_dir = os.path.dirname(progress_file)
progress_filename = os.path.basename(progress_file)


model = sys.argv[1]
num_batches = int(sys.argv[2])
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])


def docker_image():
    print("Building Docker image...")
    try:
        subprocess.run(
            f"docker build -t {IMAGE} "
            f"--cpu-quota={int(100000 * MAX_CORES)} "
            f"--build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . ",
            shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("Docker image built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"FATAL: Docker build failed. Error: {e}")
        exit(1)

    try:
        docker_command = (
    f'docker run --rm -v {progress_dir}:/project/results --name={job_id} '
    f'--cpu-period=100000 --cpu-quota={int(MAX_CORES * 100000)} '
    f'{IMAGE} {model} {str(num_batches)} {str(epochs)} {str(batch_size)} {progress_filename}'
)
        result = subprocess.run(docker_command, shell=True, check=True, capture_output=True)
        print("OUTPUT dal container:")
        print(result.stdout)
        print("ERRORI dal container:")
        print(result.stderr)

    except Exception as e:
        print(e)
        print("HA FALLITO LA RUN")
        exit(1)

os.makedirs(progress_dir, exist_ok=True)
docker_image()
