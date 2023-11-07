from typing import *
import numpy as np

def compute_metrics(eval_pred):
    """Compute metrics for evaluation predictions.

    Args:
        eval_pred (tuple): A tuple containing logits and labels.

    Returns:
        float: The computed metric value.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
