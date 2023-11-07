from typing import *
from typing import List

def add_numbers(numbers: List[int]) -> int:
    '''Add up all the numbers in the given list.

    Args:
        numbers (List[int]): A list of numbers.

    Returns:
        int: The sum of all the numbers.
    '''
    sum = 0
    for num in numbers:
        sum += num
    return sum
