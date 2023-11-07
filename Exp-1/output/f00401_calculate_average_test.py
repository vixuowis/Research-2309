from f00401_calculate_average import *
import numpy as np
import pytest


def test_calculate_average():
    # Test case 1
    numbers1 = [1, 2, 3, 4, 5]
    expected1 = np.mean(numbers1)
    assert calculate_average(numbers1) == expected1
    
    # Test case 2
    numbers2 = [10, 20, 30, 40, 50]
    expected2 = np.mean(numbers2)
    assert calculate_average(numbers2) == expected2
    
    # Test case 3
    numbers3 = [2.5, 3.5, 4.5, 5.5, 6.5]
    expected3 = np.mean(numbers3)
    assert calculate_average(numbers3) == expected3
    
    # Test case 4
    numbers4 = []
    expected4 = np.mean(numbers4)
    assert calculate_average(numbers4) == expected4
    
    # Test case 5
    numbers5 = [1]
    expected5 = np.mean(numbers5)
    assert calculate_average(numbers5) == expected5
    
pytest.main(['-v', '--tb=line', '-rN', __file__])
