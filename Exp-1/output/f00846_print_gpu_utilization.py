from typing import *
import torch

def print_gpu_utilization():
    # This function prints the amount of GPU memory occupied.
    tensor = torch.ones((1, 1)).to('cuda')
    memory = torch.cuda.memory_allocated()
    print(f'GPU memory occupied: {memory} MB.')
