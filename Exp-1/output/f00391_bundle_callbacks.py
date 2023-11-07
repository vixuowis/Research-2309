from typing import *
from typing import List

def bundle_callbacks(callbacks: List[callable]) -> callable:
    """
    This function takes a list of callbacks and returns a single callback that executes all the callbacks in the list sequentially.

    Parameters:
    - callbacks: A list of callable objects representing the callbacks to be bundled together.

    Returns:
    - A callable object that executes all the callbacks in the list sequentially.
    """
    def bundled_callback(*args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)
    return bundled_callback
