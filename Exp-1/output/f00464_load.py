from typing import *
import evaluate

def load(metric_name: str) -> Any:
    """Load a metric.

    Args:
        metric_name (str): The name of the metric to load.

    Returns:
        Any: The loaded metric."""
    return evaluate.load(metric_name)
