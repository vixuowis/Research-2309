from typing import *
from transformers import TapasConfig, TFTapasForQuestionAnswering

def initialize_model():
    """
    Initialize the Tapas model with custom classification heads.

    Returns:
        model (TFTapasForQuestionAnswering): The initialized Tapas model.
    """
    config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
    model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
    return model
