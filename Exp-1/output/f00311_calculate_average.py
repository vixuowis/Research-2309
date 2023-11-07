from typing import *
import numpy as np

def calculate_average(numbers):
    """Calculate the average of a list of numbers

    Args:
        numbers (list): A list of numbers

    Returns:
        float: The average of the numbers
    """
    if len(numbers) == 0:
        return 0.0
    else:
        return np.mean(numbers)
