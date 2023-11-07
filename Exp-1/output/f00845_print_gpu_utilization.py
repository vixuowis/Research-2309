from typing import *
import torch

def print_gpu_utilization():
    """Prints the current GPU memory utilization.

    Returns:
        None."""
    torch.cuda.empty_cache()

    gpu_memory = torch.cuda.memory_allocated()
    print(f'GPU memory occupied: {gpu_memory} MB.')
