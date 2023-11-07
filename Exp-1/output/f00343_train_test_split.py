from typing import *
from datasets import Dataset

def train_test_split(dataset, test_size):
    """Split the dataset into a train and test set.

    Args:
        dataset (Dataset): The dataset to split.
        test_size (float): The proportion of the dataset to include in the test set.

    Returns:
        Tuple[Dataset, Dataset]: The train and test sets."""
    train_set, test_set = dataset.train_test_split(test_size=test_size)
    return train_set, test_set
