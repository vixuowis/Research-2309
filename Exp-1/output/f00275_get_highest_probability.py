from typing import *
import torch

def get_highest_probability(outputs):
    '''
    Get the highest probability from the model output for the start and end positions.
    
    Args:
        outputs (torch.Tensor): Model output tensor with start_logits and end_logits.
    
    Returns:
        Tuple[int, int]: Tuple containing the start and end indices with highest probabilities.
    '''
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
