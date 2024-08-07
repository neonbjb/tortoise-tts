import os

def pick_max_worker_function():
    # Determine the number of CPU cores
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1  # Default to 1 if unable to determine
    
    # Assume the tasks are I/O-bound; we can afford to have more workers
    # You might want to adjust this logic based on the nature of your tasks
    max_workers = cpu_count * 2
    print(f"Picked max workers: {max_workers}")
    return max_workers