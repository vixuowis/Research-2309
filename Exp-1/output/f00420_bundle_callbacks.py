from typing import *
from typing import List

def bundle_callbacks(callbacks: List[callable]) -> callable:
    '''
    This function takes a list of callback functions and returns a single callback function that calls each of the input callbacks in sequence.

    :param callbacks: A list of callback functions
    :return: A single callback function
    '''
    def bundled_callback(*args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)

        return bundled_callback
