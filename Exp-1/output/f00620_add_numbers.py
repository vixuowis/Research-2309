from typing import *
from typing import List

def add_numbers(numbers: List[int]) -> int:
    """Add up all the numbers in the given list.

    Args:
        numbers (List[int]): A list of integers.

    Returns:
        int: The sum of all the numbers in the list.
    """
    result = 0

    for num in numbers:
        result += num

    return result
