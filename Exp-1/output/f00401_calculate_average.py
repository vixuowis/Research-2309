from typing import *
import numpy as np

def calculate_average(numbers):
    # Calculate the sum of the numbers
    total = sum(numbers)
    
    # Calculate the average
    average = total / len(numbers)
    
    # Return the average
    return average
