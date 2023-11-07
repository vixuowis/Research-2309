from typing import *
import evaluate

def load(metric_name: str) -> Any:
    """Load the specified metric.

    Args:
        metric_name (str): The name of the metric to load.

    Returns:
        Any: The loaded metric object.
    """
    return evaluate.load(metric_name)
