from typing import *
from datasets import Dataset

def train_test_split(test_size: float) -> Tuple[Dataset, Dataset]:
    """Split the dataset into a train and test set.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        Tuple[Dataset, Dataset]: The train and test datasets."""
    train_dataset, test_dataset = billsum.train_test_split(test_size=0.2)
    return train_dataset, test_dataset
