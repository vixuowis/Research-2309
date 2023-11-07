from typing import *
import numpy as np

def calculate_average(numbers):
    # Check if the list is empty
    if len(numbers) == 0:
        return None
    
    # Calculate the average
    average = np.mean(numbers)
    
    return average
