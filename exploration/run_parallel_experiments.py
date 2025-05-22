import subprocess
import concurrent.futures
import queue
import threading
import logging
from datetime import datetime
import time
import random
import medmnist

# List of base terminal commands (without --gpu argument)
INFO = medmnist.INFO
# Get all datasets
base_commands = [
    'python3 train.py --data=MNIST --dataset=MultiWMNIST --x2_angle=60 --latent_dim=30 --run_id=-1 --model=MultiWFunctorMNIST'
]
commands = base_commands * 8

# Setup logging to a text file
log_file = "execution.log"
job_delay = 5

def log_event(event, command, gpu_id):
    """Logs execution start/finish events to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - [GPU {gpu_id}] {event}: {command}\n"
    
    # Append log to the file
    with open(log_file, "a") as log:
        log.write(log_message)
    
    print(log_message.strip())  # Also print to console



num_gpus = 4  # Number of GPUs available
N = 2         # Number of concurrent processes per GPU

# Create a shared job queue
job_queue = queue.Queue()
for cmd in commands:
    job_queue.put(cmd)

# Track how many jobs are running on each GPU
gpu_locks = [threading.Semaphore(N) for _ in range(num_gpus)]
# Global timestamp for last job execution
last_exec_time = 0
time_lock = threading.Lock()  # Ensures safe updates to last_exec_time

def run_command(command, gpu_id):
    """Run a shell command with GPU ID injection and enforce global delay."""
    global last_exec_time

    # Ensure global delay between job launches
    with time_lock:
        current_time = time.time()
        time_since_last = current_time - last_exec_time

        if time_since_last < job_delay:
            sleep_time = job_delay - time_since_last
            print(f"Waiting {sleep_time:.2f}s before starting next job.")
            time.sleep(sleep_time)  # Wait until 5s have passed

        # Update global last execution time
        last_exec_time = time.time()

    command = f"{command} --gpu_id {gpu_id}"
    
    # Log start
    log_event("START", command, gpu_id)

    # Open a log file to capture output per command
    output_log_file = f"gpu_{gpu_id}.log"
    with open(output_log_file, "a") as out_log:
        process = subprocess.run(command, shell=True, stdout=out_log, stderr=out_log)

    # Log finish
    log_event("FINISH", command, gpu_id)

    return process.returncode

def gpu_worker(gpu_id):
    """Worker function that dynamically picks jobs and assigns GPU IDs."""
    while not job_queue.empty():
        gpu_locks[gpu_id].acquire()  # Ensure max N parallel jobs per GPU
        
        try:
            command = job_queue.get_nowait()  # Get a job from the queue
        except queue.Empty:
            gpu_locks[gpu_id].release()  # Release GPU slot if queue is empty
            break

        future = executor.submit(run_command, command, gpu_id)

        def job_done_callback(fut):
            gpu_locks[gpu_id].release()  # Free GPU slot when done

        future.add_done_callback(job_done_callback)

# Global executor for managing subprocesses
executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus * N)

# Start GPU workers dynamically
with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as gpu_executor:
    futures = {gpu_executor.submit(gpu_worker, gpu_id): gpu_id for gpu_id in range(num_gpus)}

    for future in concurrent.futures.as_completed(futures):
        gpu_id = futures[future]
        try:
            future.result()
        except Exception as e:
            log_event("ERROR", f"GPU {gpu_id} encountered an error: {e}", gpu_id)

print(f"All commands executed. Check {log_file} for execution details.")
print(f"Per-GPU logs: gpu_0.log, gpu_1.log, gpu_2.log, gpu_3.log")
with open(log_file, "a") as log:
    log.write(f"All jobs finished\n")