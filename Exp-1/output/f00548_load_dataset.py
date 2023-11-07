from typing import *
from datasets import load_dataset


def load_dataset(dataset_name: str, split: str = 'train') -> Dataset:
    """Loads a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load. Defaults to 'train'.

    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset
