from typing import *
from datasets import load_dataset

def load_dataset(dataset_name: str, split: str) -> Dataset:
    """Load a dataset from the datasets library.

    Args:
        dataset_name (str): The name of the dataset.
        split (str): The split of the dataset to load.

    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split)
    return dataset
