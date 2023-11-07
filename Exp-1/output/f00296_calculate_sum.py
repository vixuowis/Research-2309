from typing import *
from typing import List

def calculate_sum(numbers: List[int]) -> int:
    """Calculates the sum of a list of numbers.

    Args:
        numbers (List[int]): A list of integers.

    Returns:
        int: The sum of the numbers.
    """
    # Initialize the sum to 0
    total = 0

    # Iterate over each number in the list
    for num in numbers:
        # Add the number to the sum
        total += num

    # Return the final sum
    return total
