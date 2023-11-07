from f00341_calculate_average import *
import numpy as np
import pytest


def test_calculate_average():
    # Test case 1: Empty list
    assert calculate_average([]) is None
    
    # Test case 2: List with one number
    assert calculate_average([5]) == 5
    
    # Test case 3: List with multiple numbers
    numbers = [1, 2, 3, 4, 5]
    assert calculate_average(numbers) == np.mean(numbers)
    
    # Test case 4: List with negative numbers
    numbers = [-1, -2, -3, -4, -5]
    assert calculate_average(numbers) == np.mean(numbers)
    
    # Test case 5: List with decimal numbers
    numbers = [1.5, 2.5, 3.5, 4.5, 5.5]
    assert calculate_average(numbers) == np.mean(numbers)
