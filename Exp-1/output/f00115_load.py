from typing import *
import evaluate


def load(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Load a metric function from the evaluate library.
    
    Args:
        - metric (str): The name of the metric function to load.
    
    Returns:
        - Callable[[np.ndarray, np.ndarray], float]: The loaded metric function.
    """
    return evaluate.load(metric)
