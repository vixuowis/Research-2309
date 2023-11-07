from typing import *
from datasets import load_dataset, Audio

def load_voxpopuli_dataset(language: str) -> Audio:
    '''
    Load the VoxPopuli dataset for a specific language.

    Args:
        language (str): The language code of the dataset subset to load.

    Returns:
        Audio: The loaded VoxPopuli dataset.
    '''
    dataset = load_dataset('facebook/voxpopuli', language, split='train')
    return dataset
