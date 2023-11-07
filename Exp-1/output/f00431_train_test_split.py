from typing import *
from datasets import Dataset

def train_test_split(dataset, test_size):
    """
    Split the dataset's train split into a smaller train and test set.

    Args:
    - dataset (Dataset): The dataset to split.
    - test_size (float): The proportion of the dataset to include in the test split. Should be between 0 and 1.

    Returns:
    - train_dataset (Dataset): The smaller train dataset.
    - test_dataset (Dataset): The test dataset.
    """
    train_dataset = dataset.train_test_split(test_size=test_size)['train']
    test_dataset = dataset.train_test_split(test_size=test_size)['test']
    return train_dataset, test_dataset
