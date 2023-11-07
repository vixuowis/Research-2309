from typing import *
import torch

def run_benchmark():
    """
    Runs the benchmark on the instantiated benchmark object.

    Returns:
        str: The benchmark results.
    """
    results = benchmark.run()
    print(results)
