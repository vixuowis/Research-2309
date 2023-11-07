from typing import *
from typing import List

def add_numbers(numbers: List[int]) -> int:
    """Add all the numbers in the given list.

    Args:
        numbers (List[int]): A list of integers.

    Returns:
        int: The sum of all the numbers.
    """
    # Initialize the sum to 0
    sum = 0

    # Iterate through the numbers and add each number to the sum
    for num in numbers:
        sum += num

    # Return the sum
    return sum
