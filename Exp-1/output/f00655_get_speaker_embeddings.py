from typing import *
import numpy as np

def get_speaker_embeddings(processed_example):
    '''
    Calculate the speaker embeddings from the processed example.

    Args:
        processed_example (dict): A dictionary containing the processed example.

    Returns:
        numpy.ndarray: The speaker embeddings as a 512-element vector.
    '''
    speaker_embeddings = processed_example["speaker_embeddings"]
    return np.array(speaker_embeddings)
