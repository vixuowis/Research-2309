from typing import *
from transformers import TapasConfig, TapasForQuestionAnswering

def initialize_model(num_aggregation_labels: int, average_logits_per_cell: bool) -> TapasForQuestionAnswering:
    """Initialize a TapasForQuestionAnswering model with custom classification heads.

    Args:
        num_aggregation_labels (int): The number of aggregation labels.
        average_logits_per_cell (bool): Whether to average the logits per cell.

    Returns:
        model (TapasForQuestionAnswering): The initialized model.
    """
    config = TapasConfig(num_aggregation_labels=num_aggregation_labels, average_logits_per_cell=average_logits_per_cell)
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
    return model
