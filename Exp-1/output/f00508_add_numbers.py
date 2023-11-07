from typing import *
from typing import List

def add_numbers(numbers: List[int]) -> int:
    # This function takes a list of numbers as input and returns the sum of all the numbers.
    
    # Initialize the sum
    total = 0

    # Iterate over the numbers
    for num in numbers:
        # Add each number to the sum
        total += num

    # Return the sum
    return total
