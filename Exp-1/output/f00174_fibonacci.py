from typing import *
from typing import List

def fibonacci(n: int) -> List[int]:
    """
    Calculate the Fibonacci sequence up to the given number.

    Args:
        n (int): The number up to which the Fibonacci sequence should be calculated.

    Returns:
        List[int]: The Fibonacci sequence up to the given number.
    """
    fib = [0, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    return fib[:-1]
