from typing import *
from typing import List

def add_numbers(numbers: List[int]) -> int:
    """
    This function takes a list of numbers as input and returns the sum of all the numbers.

    Args:
        numbers (List[int]): A list of numbers.

    Returns:
        int: The sum of all the numbers.
    """
    # Initialize the sum
    total = 0

    # Iterate over the numbers and add them to the sum
    for num in numbers:
        total += num

    # Return the sum
    return total
