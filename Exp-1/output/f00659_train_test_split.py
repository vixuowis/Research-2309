from typing import *
from sklearn.model_selection import train_test_split

def train_test_split(test_size):
    """Split the dataset into train and test sets.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing the train and test sets.
    """
    train_set, test_set = train_test_split(dataset, test_size=test_size)
    return train_set, test_set
