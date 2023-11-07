from typing import *
from typing import List

def multiply(nums: List[int]) -> int:
    """
    Multiply all the numbers in the given list.

    Args:
        nums (List[int]): A list of integers.

    Returns:
        int: The product of all the numbers in the list.
    """
    product = 1
    for num in nums:
        product *= num
    return product
