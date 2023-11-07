from typing import *
import pandas as pd

def count_speakers(dataset):
    '''
    Count the number of unique speakers in the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset containing the speaker IDs.

    Returns:
        int: The number of unique speakers.
    '''
    return len(set(dataset['speaker_id']))
