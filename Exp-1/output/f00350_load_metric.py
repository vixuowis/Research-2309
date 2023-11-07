from typing import *
import evaluate

def load_metric(metric_name: str) -> Metric:
    """Load a metric by name.

    Args:
        metric_name (str): The name of the metric to load.

    Returns:
        Metric: The loaded metric."""
    metric = evaluate.load(metric_name)
    return metric
