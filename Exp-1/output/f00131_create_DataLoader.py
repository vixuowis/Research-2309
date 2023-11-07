from typing import *
from torch.utils.data import DataLoader

def create_DataLoader(dataset, shuffle=False, batch_size=1):
    '''
    Create a DataLoader for the given dataset.

    Args:
        dataset: The dataset to create DataLoader for.
        shuffle: Whether to shuffle the data (default: False).
        batch_size: Number of samples per batch (default: 1).

    Returns:
        DataLoader: The created DataLoader.
    '''
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
